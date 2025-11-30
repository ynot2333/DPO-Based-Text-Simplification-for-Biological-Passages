


# train_dpo_fp16_lora_multi_fixed.py

import os
import json
import random
import torch
import numpy as np
from datasets import Dataset, DatasetDict

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig


# --------------------- 基本配置 ---------------------
MODEL = "deepseek-ai/DeepSeek-V2-Lite-chat"
DATAFILE = "datasets_20251123.json"
REF_LOGPROBS_DIR = "ref_logprobs"
OUTPUT_DIR = "dpo_fp16_lora_out_v2"

SCORE_DIFF_THRESH = 0.1

# LoRA 超参
LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# 训练超参（修改后）
PER_DEVICE_BATCH_SIZE = 3
GRAD_ACC = 4
LR = 5e-6
MAX_STEPS = 600

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# --------------------- Dataset 构建 ---------------------
class PairDatasetBuilder:
    def __init__(self, data_path, ref_dir, score_diff):
        self.ref_dir = ref_dir
        self.score_diff = score_diff
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def build(self):
        pairs = []
        for key, item in self.data.items():
            sims = item["simplifications"]

            tries = 0
            while True:
                i, j = random.sample(range(len(sims)), 2)
                if abs(sims[i]["score"] - sims[j]["score"]) > self.score_diff:
                    break
                tries += 1
                if tries > 50:  # fallback
                    scores = [s["score"] for s in sims]
                    i = int(np.argmax(scores))
                    j = int(np.argmin(scores))
                    break

            if sims[i]["score"] > sims[j]["score"]:
                chosen_idx, rejected_idx = i, j
            else:
                chosen_idx, rejected_idx = j, i

            
            prompt_words = '我是一名高中生，请把以下文本简化一下，让高中生可以读，严格输出简化后的中文文本，不要有任何解释说明只要结果。文本内容：'

            pairs.append({
                "id": key,
                "prompt": prompt_words+item["paragraph"],
                "chosen": sims[chosen_idx]["text"],
                "rejected": sims[rejected_idx]["text"],
                "chosen_idx": chosen_idx,
                "rejected_idx": rejected_idx,
                "ref_path": os.path.join(self.ref_dir, f"{key}.pt"),
            })
        return pairs


# --------------------- Tokenizer & Model ---------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

# --------------------- Apply LoRA ---------------------
peft_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()


# --------------------- 自定义 DPOTrainer：加载 ref_logprobs ---------------------
class MyDPOTrainer(DPOTrainer):
    def compute_reference_log_probs(self, batch):
        chosen_ref, rejected_ref = [], []
        for ref_path, ci, ri in zip(batch["ref_path"], batch["chosen_idx"], batch["rejected_idx"]):
            data = torch.load(ref_path, map_location="cpu")
            arr = data["candidate_logprobs"]
            arr = arr if isinstance(arr, list) else arr.tolist()
            chosen_ref.append(arr[int(ci)])
            rejected_ref.append(arr[int(ri)])

        device = next(self.model.parameters()).device
        return (
            torch.tensor(chosen_ref, dtype=torch.float32, device=device),
            torch.tensor(rejected_ref, dtype=torch.float32, device=device)
        )


# --------------------- 构建 Dataset 并划分 1200/92 ---------------------
builder = PairDatasetBuilder(DATAFILE, REF_LOGPROBS_DIR, SCORE_DIFF_THRESH)
pairs_list = builder.build()

random.shuffle(pairs_list)

train_data = pairs_list[:-92]
valid_data = pairs_list[-92:]

ds = DatasetDict({
    "train": Dataset.from_list(train_data),
    "validation": Dataset.from_list(valid_data),
})

print(f"Train: {len(train_data)} pairs, Validation: {len(valid_data)} pairs.")


# --------------------- 配置 DPOConfig ---------------------
dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,

    max_steps=MAX_STEPS,
    learning_rate=LR,

    warmup_steps=50,
    logging_steps=20,

    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,

    bf16=True,
    remove_unused_columns=False,
    report_to="none",

    # ⭐ 重要：提高 KL 权重（防过拟合）
    beta=0.3,
    loss_type="sigmoid",
)


# --------------------- Callback：简单 eval + early stopping ---------------------
class SimpleEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, patience=2):
        self.tokenizer = tokenizer
        self.patience = patience
        self.best_loss = float("inf")
        self.bad_epochs = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        val_loss = metrics.get("eval_loss", None)
        if val_loss is None:
            return

        print(f"[Eval] step {state.global_step} - val_loss = {val_loss:.4f}")

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            print(f"no improvement for {self.bad_epochs} eval rounds")

        if self.bad_epochs >= self.patience:
            print("[EarlyStop] Validation loss stopped improving.")
            control.should_training_stop = True


# --------------------- 构建 Trainer ---------------------
trainer = MyDPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tokenizer,
    callbacks=[SimpleEvalCallback(tokenizer)],
)


# --------------------- Run ---------------------
if __name__ == "__main__":
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print("DPO + LoRA training completed.")

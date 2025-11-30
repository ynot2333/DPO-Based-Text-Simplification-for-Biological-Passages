# # precompute_ref_logprobs.py
# """
# 用途：为每条样本的每个候选回答保存 reference 模型对该 (prompt + candidate) 的 sequence log-prob
# 输出格式（torch.save）：
# {
#   "id": "<sample_id>",
#   "candidate_logprobs": [ float, float, ... ],             # sequence log-prob (sum of token log-probs)
#   "candidate_token_logprobs": [ [float,...], ... ]        # 可选：每个 token 的 log-prob (可省略以节省空间)
# }
# """
# import os
# import torch
# from datasets import load_dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# from tqdm import tqdm
# import math
# import json

# MODEL = "deepseek-ai/DeepSeek-V2-Lite"  # 替换成你模型的 repo id
# SAVE_DIR = "ref_logprobs"
# os.makedirs(SAVE_DIR, exist_ok=True)

# device = "cuda:0"

# # bitsandbytes 配置，加载为 4-bit 以节省显存（参考 QLoRA）
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
# # 以 4-bit 加载 reference 模型（只用于一次性预计算）
# ref_model = AutoModelForCausalLM.from_pretrained(
#     MODEL,
#     quantization_config=bnb_config,
#     # device_map="auto",
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True
# ).to(device)
# ref_model.eval()

# def compute_sequence_logprob(prompt, candidate, return_token_logprobs=False):
#     # 构造输入： prompt + candidate 作为一个完整序列来计算 conditional log-probs
#     # 注意：为了得到 p(candidate | prompt) 我们需要只对 candidate 部分计算 log-probs 条件于 prompt
#     # 方法：对 prompt+cand 整体做一次 forward，取 candidate tokens 的 log-probs 条目并求和
#     text = prompt + tokenizer.eos_token + candidate
#     enc = tokenizer(text, return_tensors="pt", truncation=True).to(device)
#     input_ids = enc["input_ids"]
#     attention_mask = enc["attention_mask"]

#     with torch.no_grad():
#         outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask)
#         logits = outputs.logits  # (1, seq_len, vocab)
#         # 计算 log_softmax 得到 log-probs
#         log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (1, seq_len, vocab)

#     # 找到 candidate 在 input_ids 中的范围：先用 tokenizer to get prompt length
#     prompt_ids = tokenizer(prompt + tokenizer.eos_token, return_tensors="pt")["input_ids"]
#     prompt_len = prompt_ids.size(1)
#     seq_len = input_ids.size(1)
#     # candidate tokens 开始于 prompt_len，直到 seq_len-1
#     candidate_token_ids = input_ids[0, prompt_len:].tolist()
#     if len(candidate_token_ids) == 0:
#         # 可能 candidate 为空（不太可能），返回 -inf
#         return float("-inf"), [] if return_token_logprobs else float("-inf")

#     # 对 candidate 每个 token 取对应 log-prob
#     token_logps = []
#     for i, tok in enumerate(candidate_token_ids):
#         idx = prompt_len + i
#         token_logp = log_probs[0, idx - 1, tok].item() if idx - 1 >= 0 else log_probs[0, idx, tok].item()
#         # careful: language models predict token_{t} given inputs up to token_{t-1}
#         # above we index logits at position idx-1 to get token idx probability; but depending on tokenizer/eos we check bounds:
#         token_logps.append(token_logp)

#     # sequence log-prob = sum token_logps
#     seq_logprob = float(sum(token_logps))
#     token_logps = [float(x) for x in token_logps]
#     return seq_logprob, token_logps if return_token_logprobs else seq_logprob

# def main():

#     with open("datasets_20251123.json", "r", encoding="utf-8") as f:
#         data = json.load(f)
    
#     print(f"总共 {len(data)} 个样本")
    
#     # 处理每个样本
#     for sample_id, sample_data in tqdm(data.items(), desc="Processing samples"):
#         try:
#             # 提取prompt和candidates
#             # 根据你的数据结构，prompt可能是paragraph字段
#             prompt = sample_data["paragraph"]
            
#             # candidates是simplifications中的text字段
#             candidates = [simplification["text"] for simplification in sample_data["simplifications"]]
            
#             out = {
#                 "id": sample_id,
#                 "name": sample_data.get("name", ""),  # 保存名称信息
#                 "candidate_logprobs": [],
#                 "candidate_token_logprobs": []
#             }
            
#             # 计算每个候选回答的对数概率
#             for cand in candidates:
#                 seq_lp, token_lps = compute_sequence_logprob(
#                     prompt, cand, return_token_logprobs=True
#                 )
#                 out["candidate_logprobs"].append(seq_lp)
#                 out["candidate_token_logprobs"].append(token_lps)
            
#             # 保存结果
#             save_path = os.path.join(SAVE_DIR, f"{sample_id}.pt")
#             torch.save(out, save_path)
            
#         except Exception as e:
#             print(f"Error processing sample {sample_id}: {e}")
#             continue

#     print("All done. Reference logprobs saved to", SAVE_DIR)

# if __name__ == "__main__":
#     main()



'''

version 2
'''
# precompute_ref_logprobs_int8.py
"""
用途：为每条样本的每个候选回答保存 reference 模型对 (prompt + candidate) 的 sequence log-prob
输出格式（torch.save）：
{
  "id": "<sample_id>",
  "candidate_logprobs": [ float, ... ],
  "candidate_token_logprobs": [ [float,...], ... ]
}
"""

import os
import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "deepseek-ai/DeepSeek-V2-Lite"
SAVE_DIR = "ref_logprobs"
os.makedirs(SAVE_DIR, exist_ok=True)

device = "cuda:0"

# --------------------------
#    Tokenizer
# --------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    use_fast=False,
    trust_remote_code=True
)

# --------------------------
#    INT8 模型加载 (HF accelerate)
#    —— 无 bitsandbytes！
# --------------------------

# ref_model = AutoModelForCausalLM.from_pretrained(
#     MODEL,
#     load_in_8bit=True,        # 开启 INT8 权重量化
#     device_map={"": device},  # 全部放在 GPU0
#     trust_remote_code=True
# )

ref_model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
ref_model.eval()


# ----------------------------------------------
#             计算 sequence log-prob
# ----------------------------------------------
def compute_sequence_logprob(prompt, candidate, return_token_logprobs=False):

    # prompt + eos + candidate
    text = prompt + tokenizer.eos_token + candidate
    enc = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    with torch.no_grad():
        outputs = ref_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits  # (1, seq_len, vocab)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # prompt token 个数
    prompt_ids = tokenizer(
        prompt + tokenizer.eos_token, return_tensors="pt"
    )["input_ids"]
    prompt_len = prompt_ids.size(1)
    seq_len = input_ids.size(1)

    # candidate tokens
    candidate_token_ids = input_ids[0, prompt_len:].tolist()

    token_logps = []
    for i, tok in enumerate(candidate_token_ids):
        idx = prompt_len + i
        # 预测 token_t 来自 logits[t-1]
        token_logp = log_probs[0, idx - 1, tok].item()
        token_logps.append(token_logp)

    seq_logprob = float(sum(token_logps))
    token_logps = [float(x) for x in token_logps]

    return (seq_logprob, token_logps) if return_token_logprobs else seq_logprob


# ----------------------------------------------
#                    主程序
# ----------------------------------------------
def main():

    # 加载你的数据 json
    with open("datasets_20251123.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"总共 {len(data)} 个样本")

    for sample_id, sample_data in tqdm(data.items(), desc="Processing samples"):

        try:
            prompt = sample_data["paragraph"]
            candidates = [x["text"] for x in sample_data["simplifications"]]

            out = {
                "id": sample_id,
                "name": sample_data.get("name", ""),
                "candidate_logprobs": [],
                "candidate_token_logprobs": []
            }

            for cand in candidates:
                seq_lp, token_lps = compute_sequence_logprob(
                    prompt, cand, return_token_logprobs=True
                )
                out["candidate_logprobs"].append(seq_lp)
                out["candidate_token_logprobs"].append(token_lps)

            torch.save(out, os.path.join(SAVE_DIR, f"{sample_id}.pt"))

        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            continue

    print("Done! Saved to", SAVE_DIR)


if __name__ == "__main__":
    main()

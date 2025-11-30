


import os
import json
import time
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
from peft import PeftModel

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

from openai import OpenAI
from requests.exceptions import RequestException



# ----------------------------
# DeepSeek API 初始化
# ----------------------------
client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

# ----------------------------
# text2vec + SentenceTransformer
# ----------------------------
model_name = "shibing624/text2vec-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model_sentence = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)

# ----------------------------
# 高中生词汇
# ----------------------------
hs_emb_path = 'datasets/textbook_resource/hs_emb.npy'
hs_words_dict_path = 'datasets/textbook_resource/texts_words.json'

hs_words_dict = json.load(open(hs_words_dict_path, 'r', encoding='utf-8'))
hs_words = [w for w in hs_words_dict.keys() if hs_words_dict[w] >= 2]
hs_emb = np.load(hs_emb_path)
threshold = 0.80

# ----------------------------
# Prompt 定义
# ----------------------------
prompt_translate = "把文本翻译成中文，直接输出翻译后的中文文本。"
prompt_easier = "我是一名高中生，请把以下文本简化一下, 严格输出简化后的中文文本，不要有任何解释说明只要结果。文本内容："

# ----------------------------
# 工具函数
# ----------------------------
def retry_request(func, retries=3, delay=2, *args, **kwargs):
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"请求失败: {e}, 尝试重试 {attempt+1}/{retries}")
            time.sleep(delay)
    return None

def dpo_process(prompt):

    inputs = tokenizer_dpo(prompt, return_tensors="pt").to(model_dpo.device)

    # 生成
    outputs = model_dpo.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )

    return ''.join(tokenizer_dpo.decode(outputs[0], skip_special_tokens=True).strip().split("\n")[1:])




def query_deepseek_chgtext(text, prompt_input):
    prompt = f"{prompt_input}{text}"
    def request():
        return client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        ).choices[0].message.content.strip()
    return retry_request(request)

def query_deepseek_words(text):
    prompt = f"请从文本中提取每一个生物学相关中文名词，直接输出最终名词结果，名词之间用英文逗号间隔。{text}"
    def request():
        result = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        ).choices[0].message.content.strip()
        return result.split(',')
    return retry_request(request)

def query_deepseek_completeness(target_text, rewrite_text):
    prompt = f"""
给定两个句子 A 和 B，请判断：
1. B 是否删除了 A 中的关键信息？
2. B 覆盖 A 的语义比例是多少？（0~1 表示）
3. 列出 B 未覆盖的信息点。

严格输出 2中completedness的值:

句子A: {target_text}
句子B: {rewrite_text}
"""
    def request():
        result = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        ).choices[0].message.content.strip()
        return result
    return retry_request(request)


def cosine_similarity(A, B):
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    return np.dot(A_norm, B_norm.T)

def embed(words):
    inputs = tokenizer(words, padding=True, truncation=True, return_tensors="pt").to('cuda:1')
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state[:, 0, :]
    return outputs.cpu().numpy()

def meaning_similarity(text1, text2):
    emb = model_sentence.encode([text1, text2])
    return cosine_similarity([emb[0]], [emb[1]])[0][0]

def text_construct(text, prompt):
    text_dict = {}
    prompt_dpo = prompt + text
    text_e = dpo_process(prompt_dpo)
    if '我是一名高中生' in text_e:
        return None
    if not text_e:
        return None
    query_e = query_deepseek_words(text_e)
    if not query_e:
        query_e = []
    query_e_emb = embed(query_e) if query_e else np.zeros((1, hs_emb.shape[1]))
    sim_matrix = cosine_similarity(query_e_emb, hs_emb) if query_e else np.zeros((1, hs_emb.shape[0]))
    max_sim = sim_matrix.max(axis=1) if query_e else np.array([0])
    results_e = max_sim > threshold
    ave_e = max_sim.mean()
    true_e = results_e.sum()
    ms = meaning_similarity(text, text_e)
    completeness = query_deepseek_completeness(text, text_e)
    text_dict['text_e'] = text_e
    text_dict['text_e_meaning_similarity'] = str(ms)
    text_dict['text_e_words'] = query_e
    text_dict['text_e_completeness'] = str(completeness)
    text_dict['text_e_hs_pass_rate'] = str(true_e/len(results_e))
    text_dict['text_e_hs_score'] = str(ave_e)
    return text_dict

# ----------------------------
# 处理单个文档
# ----------------------------
def process_document(text_ori):
    # try:
    text = query_deepseek_chgtext(text_ori, prompt_translate)
    if not text:
        return None
    # 高中生词汇相似度
    query = query_deepseek_words(text) or []

    query_emb = embed(query) if query else np.zeros((1, hs_emb.shape[1]))
    sim_matrix = cosine_similarity(query_emb, hs_emb) if query else np.zeros((1, hs_emb.shape[0]))
    max_sim = sim_matrix.max(axis=1) if query else np.array([0])
    results = max_sim > threshold
    ave = max_sim.mean()
    true = results.sum()
    
    # 构建文本处理
    text_dict = text_construct(text, prompt_easier)

    print(f'原文翻译结果：{text}')
    print(f'原文生物名词：{query}')
    print(f'原文可读性（0-1）：{str(true/len(results))}')
    print(f'模型简化结果：{text_dict["text_e"]}')
    print(f'模型简化生物名词：{text_dict["text_e_words"]}')
    print(f'简化可读性（0-1）：{text_dict["text_e_hs_pass_rate"]}')
    print(f'简化语义相似性（0-1）：{text_dict["text_e_meaning_similarity"]}')
    print(f'简化语义完整性（0-1）：{text_dict["text_e_completeness"]}')


    return {
        'txt_ori': text_ori,
        'txt_translate': text,
        'txt_words': query,
        'txt_hs_pass_rate': str(true/len(results)) if results.size > 0 else '0',
        'txt_hs_score': str(ave),
        'txt': text_dict
    }
    # except Exception as e:
    #     print(f"处理文件 {file_path} 失败: {e}")
    #     return None

# ----------------------------
# 主程序
# ----------------------------
if __name__ == "__main__":

    # dpo model input
    base_model_name = "deepseek-ai/DeepSeek-V2-Lite-Chat"  # 或 DeepSeek-V2-Lite-Instruct
    lora_path = 'dpo_fp16_lora_out_v2/'
    # 加载 tokenizer
    tokenizer_dpo = AutoTokenizer.from_pretrained(base_model_name)

    if tokenizer_dpo.pad_token is None:
        tokenizer_dpo.pad_token = tokenizer_dpo.eos_token

    # 加载模型（BF16，可降低显存）
    model_dpo = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # 自动使用 GPU
        trust_remote_code=True
    )
    
    model_dpo = PeftModel.from_pretrained(
        model_dpo,
        lora_path,
    )

    model_dpo.eval()

     # 交互式输入
    print("=== 文档处理系统 ===")
    print("输入 'quit' 或 '退出' 结束程序")
    
    while True:
        # 获取用户输入
        text_ori = input("\n请输入要处理的文档内容: ").strip()
        
        # 检查退出条件
        if text_ori.lower() in ['quit', '退出', 'exit']:
            print("程序结束，再见！")
            break
            
        # 检查空输入
        if not text_ori:
            print("输入不能为空，请重新输入！")
            continue
            
        # 处理文档
        print("\n处理结果：")
        try:
            result = process_document(text_ori)
            print(result)
        except Exception as e:
            print(f"处理过程中出现错误: {e}")
    



# from transformers import AutoTokenizer, AutoModel
# from sentence_transformers import SentenceTransformer
# import torch
# import numpy as np
# import json
# import os
# from openai import OpenAI
# from tqdm import tqdm
# import sys
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('-n', '--number', type=str, help='参数')
# args = parser.parse_args()


# ''' deepseek '''
# # 设置 API Key
# os.environ['DEEPSEEK_API_KEY'] = 'sk-00244d1cacc94a388c9823ef2843292c'

# # 初始化 DeepSeek 客户端
# client = OpenAI(
#     api_key=os.environ.get('DEEPSEEK_API_KEY'),
#     base_url="https://api.deepseek.com"
# )

# ''' text2vec '''
# model_name = "shibing624/text2vec-base-chinese"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)

# model_sentence = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# def query_deepseek_words(text):
#     """处理单个文本段落：调用 deepseek API，提取名词"""

#     prompt = f"""
# 请从文本中提取每一个生物学相关中文名词，直接输出最终名词结果，名词之间用英文逗号间隔。{text}
# """
#     try:
#         response = client.chat.completions.create(
#             model="deepseek-chat",
#             messages=[{"role": "user", "content": prompt}],
#             stream=False,
#         )
#         result = response.choices[0].message.content.strip()
#         return result.split(',')
#     except Exception as e:
#         print(f"Error: {e}")
#         return None
    
# def query_deepseek_chgtext(text, prompt_input):

#     prompt = f"""{prompt_input}{text}"""
#     try:
#         response = client.chat.completions.create(
#             model="deepseek-chat",
#             messages=[{"role": "user", "content": prompt}],
#             stream=False,
#         )
#         result = response.choices[0].message.content.strip()
#         return result
#     except Exception as e:
#         print(f"Error: {e}")
#         return None
    
# def query_deepseek_completeness(target_text, rewrite_text):

#     prompt = f"""
# 给定两个句子 A 和 B，请判断：
# 1. B 是否删除了 A 中的关键信息？
# 2. B 覆盖 A 的语义比例是多少？（0~1 表示）
# 3. 列出 B 未覆盖的信息点。

# 严格输出 2中completedness的值:

# 句子A: {target_text}
# 句子B: {rewrite_text}
# """
#     try:
#         response = client.chat.completions.create(
#             model="deepseek-chat",
#             messages=[{"role": "user", "content": prompt}],
#             stream=False,
#         )
#         result = response.choices[0].message.content.strip()
#         return result
#     except Exception as e:
#         print(f"Error: {e}")
#         return None


# def cosine_similarity(A, B):
#     """计算余弦相似度矩阵 A × B"""
#     A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
#     B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
#     return np.dot(A_norm, B_norm.T)


# def embed(words):
#     """返回 CLS 句向量（numpy 格式）"""
#     inputs = tokenizer(words, padding=True, truncation=True, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(**inputs).last_hidden_state[:, 0, :]  # 取 CLS token
#     return outputs.cpu().numpy()

# def meaning_similarity(target_text, rewrite_text):

#     '''计算语义相似度'''

#     embeddings = model_sentence.encode([target_text, rewrite_text])
#     similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

#     return similarity


# ### prompt ###

# prompt_chinese = f"""
# 把文本翻译成中文，直接输出翻译后的中文文本。
# """
# prompt_easier1 = f"""
# 我是一名高中生，请把文本简化一下，让高中生可以读，严格输出简化后的中文文本，不要有任何解释说明只要结果。
# """
# prompt_easier2 = f"""
# 我是一名高中生，请把文本简化一下，减少专有名词，严格输出简化后的中文文本，不要有任何解释说明只要结果。
# """
# prompt_easier3 = f"""
# 我是一名高中生，请概括一下文本，减少专有名词，严格输出简化后的中文文本，不要有任何解释说明只要结果。
# """


# def text_construct(text, pmpt):

#     text_dict = {}

#     text_e = query_deepseek_chgtext(text, pmpt)
#     query_e = query_deepseek_words(text_e)
#     query_e_emb = embed(query_e)
#     sim_matrix_e = cosine_similarity(query_e_emb, hs_emb)
#     max_sim_e = sim_matrix_e.max(axis=1)
#     results_e = max_sim_e > threshold
#     ave_e = max_sim_e.mean()
#     true_e = results_e.sum()
#     ms = meaning_similarity(text, text_e)
#     completeness = query_deepseek_completeness(text, text_e)

#     text_dict['text_e'] = text_e
#     text_dict['text_e_meaning_similarity'] = str(ms)
#     text_dict['text_e_completeness'] = str(completeness)
#     text_dict['text_e_hs_pass_rate'] = str(true_e/len(results_e))
#     text_dict['text_e_hs_score'] = str(ave_e)

#     return text_dict


# # 1. 高中生物词汇列表
# hs_words_dict = json.load(open('/home/zhangry/LLM-finetune/text_resource/texts_words.json'))
# hs_words = [i for i in hs_words_dict.keys() if hs_words_dict[i]>=2]

# # hs_emb = embed(hs_words)
# # np.save('/home/zhangry/LLM-finetune/text_resource/hs_emb.npy', hs_emb)

# hs_emb = np.load('/home/zhangry/LLM-finetune/text_resource/hs_emb.npy')

# # 2. 待判断词汇
# dir = '/home/zhangry/LLM-finetune/paperdownload/paper_directory/'
# doc_list = os.listdir(dir)
# doc_list = doc_list[int(args.number)*120:(int(args.number)+1)*120]
# # nn = int(args.number)*120
# # doc_list = [doc_list[nn]]

# threshold = 0.80

# data_dict = {}

# for idx, dc in enumerate(doc_list):
#     try: 
#         name = dc.replace('.json', '')
#         path = os.path.join(dir, dc)
#         text_ori = json.load(open(path))['abstract']
#         text = query_deepseek_chgtext(text_ori, prompt_chinese)

#         query = query_deepseek_words(text)
#         query_emb = embed(query)

#         sim_matrix = cosine_similarity(query_emb, hs_emb)

#         # 每个查询词找 “最高相似度”
#         max_sim = sim_matrix.max(axis=1)  # shape = [len(query)]

#         # 4. 阈值判断
#         threshold = 0.80
#         results = max_sim > threshold

#         ave = max_sim.mean()
#         true = results.sum()

#         # print("平均相似度：", ave)
#         # print("通过率：", true/len(results))

#         text_dict1 = text_construct(text, prompt_easier1)
#         text_dict2 = text_construct(text, prompt_easier2)
#         text_dict3 = text_construct(text, prompt_easier3)
#         text_dict4 = text_construct(text_dict1['text_e'], prompt_easier3)
#         text_dict5 = text_construct(text_dict2['text_e'], prompt_easier3)

#         text_dict = {
#             'txt_ori': text_ori, 
#             'txt_translate': text,
#             'txt_hs_pass_rate': str(true/len(results)),
#             'txt_hs_score': str(ave),
#             'txt1': text_dict1,
#             'txt2': text_dict2,
#             'txt3': text_dict3,
#             'txt4': text_dict4,
#             'txt5': text_dict5,
#         }

#         data_dict[name] = text_dict

#     except Exception as e:
#         print(e)

    

# json.dump(data_dict, open(f'datasets/dataset_{args.number}_20251112.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)





import os
import json
import time
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial

import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

from openai import OpenAI
from requests.exceptions import RequestException

from multiprocessing import Pool, cpu_count

# ----------------------------
# 参数解析
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--number', type=int, help='任务编号')
args = parser.parse_args()

# ----------------------------
# DeepSeek API 初始化
# ----------------------------
os.environ['DEEPSEEK_API_KEY'] = 'sk-00244d1cacc94a388c9823ef2843292c'  # 请填你的 API key
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
hs_emb_path = '/home/zhangry/LLM-finetune/text_resource/hs_emb.npy'
hs_words_dict_path = '/home/zhangry/LLM-finetune/text_resource/texts_words.json'

hs_words_dict = json.load(open(hs_words_dict_path, 'r', encoding='utf-8'))
hs_words = [w for w in hs_words_dict.keys() if hs_words_dict[w] >= 2]
hs_emb = np.load(hs_emb_path)
threshold = 0.80

# ----------------------------
# Prompt 定义
# ----------------------------
prompt_translate = "把文本翻译成中文，直接输出翻译后的中文文本。"
prompt_easier1 = "我是一名高中生，请把文本简化一下，让高中生可以读，严格输出简化后的中文文本，不要有任何解释说明只要结果。"
prompt_easier2 = "我是一名高中生，请把文本简化一下，减少专有名词，严格输出简化后的中文文本，不要有任何解释说明只要结果。"
prompt_easier3 = "我是一名高中生，请概括一下文本，减少专有名词，严格输出简化后的中文文本，不要有任何解释说明只要结果。"

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
    inputs = tokenizer(words, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state[:, 0, :]
    return outputs.cpu().numpy()

def meaning_similarity(text1, text2):
    emb = model_sentence.encode([text1, text2])
    return cosine_similarity([emb[0]], [emb[1]])[0][0]

def text_construct(text, prompt):
    text_dict = {}
    text_e = query_deepseek_chgtext(text, prompt)
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
    text_dict['text_e_completeness'] = str(completeness)
    text_dict['text_e_hs_pass_rate'] = str(true_e/len(results_e))
    text_dict['text_e_hs_score'] = str(ave_e)
    return text_dict

# ----------------------------
# 处理单个文档
# ----------------------------
def process_document(file_path):
    try:
        text_ori = json.load(open(file_path, 'r', encoding='utf-8'))['abstract']
        text = query_deepseek_chgtext(text_ori, prompt_translate)
        if not text_ori:
            return None
        if not text:
            return None
        # 高中生词汇相似度
        query = query_deepseek_words(text) or []
        if len(query) < 8:  # 词数过少，不处理
            return None
        query_emb = embed(query) if query else np.zeros((1, hs_emb.shape[1]))
        sim_matrix = cosine_similarity(query_emb, hs_emb) if query else np.zeros((1, hs_emb.shape[0]))
        max_sim = sim_matrix.max(axis=1) if query else np.array([0])
        results = max_sim > threshold
        ave = max_sim.mean()
        true = results.sum()
        if true/len(results) > 0.5:  # 句子过于简单，不处理
            return None
        
        # 构建文本处理
        text_dict1 = text_construct(text, prompt_easier1)
        text_dict2 = text_construct(text, prompt_easier2)
        text_dict3 = text_construct(text, prompt_easier3)
        text_dict4 = text_construct(text_dict1['text_e'], prompt_easier3) if text_dict1 else None
        text_dict5 = text_construct(text_dict2['text_e'], prompt_easier3) if text_dict2 else None
        return {
            'txt_ori': text_ori,
            'txt_translate': text,
            'txt_hs_pass_rate': str(true/len(results)) if results.size > 0 else '0',
            'txt_hs_score': str(ave),
            'txt1': text_dict1,
            'txt2': text_dict2,
            'txt3': text_dict3,
            'txt4': text_dict4,
            'txt5': text_dict5
        }
    except Exception as e:
        print(f"处理文件 {file_path} 失败: {e}")
        return None

# ----------------------------
# 主程序
# ----------------------------
if __name__ == "__main__":
    data_dir = '/home/zhangry/LLM-finetune/paperdownload/paper_directory/'
    files = sorted(os.listdir(data_dir))
    start_idx = 2160 + args.number * 240
    end_idx = start_idx + 240
    files_to_process = [os.path.join(data_dir, f) for f in files[start_idx:end_idx]]

    data_dict = {}
    for file_path in tqdm(files_to_process, desc=f'processing task{args.number}', position=int(args.number)):
        name = os.path.basename(file_path).replace('.json', '')
        result = process_document(file_path)
        if result:
            data_dict[name] = result

    output_path = f'datasets2/dataset_{args.number}_20251122.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    json.dump(data_dict, open(output_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    print(f"任务 {args.number} 完成，输出: {output_path}")

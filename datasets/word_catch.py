# import os
# from openai import OpenAI
# from tqdm import tqdm
# import json

# # 设置 API Key
# os.environ['DEEPSEEK_API_KEY'] = 'sk-7775a826f5ae4ee680170c45bc4bab50'

# # 初始化 DeepSeek 客户端
# client = OpenAI(
#     api_key=os.environ.get('DEEPSEEK_API_KEY'),
#     base_url="https://api.deepseek.com"
# )

# # 读取文本
# a = 320
# re_list = []

# with open('text1.txt', 'r', encoding='utf-8') as f:
#     lines = f.readlines()[a:]

# # 按每 30 行合并成一个段落
# text_list = []
# text = ''
# for idx, line in enumerate(lines):
#     if idx % 10 == 0:
#         if text != '':
#             text_list.append(text)
#         text = line.strip()
#     else:
#         text += line.strip()
# text_list.append(text)


# re_list = []
# error_c = 0
# # 循环处理每个段落，一次 prompt 完成两步
# for text in tqdm(text_list, desc='Processing r1'):
#     prompt = f"""
# 为以下内容恢复语序，然后从恢复后的文本中提取全部的生物学相关中文名词，
# 直接输出最终名词结果，名词之间用英文逗号间隔。{text}
# """
#     response = client.chat.completions.create(
#         model="deepseek-chat",
#         messages=[{"role": "user", "content": prompt}],
#         stream=False
#     )

#     condensed_text = response.choices[0].message.content.strip()
#     try:
#         re_list.extend(condensed_text.split(',').strip())
#     except Exception as e:
#         print(f'error text: {text}')
#         error_c += 1
    


# out_dict = {'words_re': re_list}

# json.dump(out_dict, open('text1_words.json', 'w'), indent=4)


import os
from openai import OpenAI
from tqdm import tqdm
import json
from multiprocessing import Pool, Manager, cpu_count
import time
import random

# 设置 API Key
os.environ['DEEPSEEK_API_KEY'] = 'sk-7775a826f5ae4ee680170c45bc4bab50'

# 初始化 DeepSeek 客户端
client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)


def process_text(text):
    """处理单个文本段落：调用 deepseek API，提取名词"""
    prompt = f"""
为以下内容恢复语序，然后从恢复后的文本中提取全部的生物学相关中文名词，
直接输出最终名词结果，名词之间用英文逗号间隔。{text}
"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        result = response.choices[0].message.content.strip()
        # 分割名词
        words = [w.strip() for w in result.split(",") if w.strip()]
        return words

    except Exception as e:
        # 输出错误信息
        print(f"[Error] {e}\nText: {text[:50]}...")
        return []


if __name__ == "__main__":

    # 读取文本
    with open("text5.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 每 10 行合并成一段
    text_list = []
    text = ""
    for idx, line in enumerate(lines):
        if idx % 10 == 0:
            if text:
                text_list.append(text)
            text = line.strip()
        else:
            text += line.strip()
    text_list.append(text)

    print(f"共 {len(text_list)} 段文本，将使用多进程处理。")

    # 创建进程池（最多使用 CPU 核数 - 1）
    num_proc = 12
    with Pool(processes=num_proc) as pool:
        results = list(tqdm(pool.imap(process_text, text_list), total=len(text_list)))

    # 合并结果
    re_list = []
    for words in results:
        re_list.extend(words)

    out_dict = {"words_re": re_list}

    with open("text5_words.json", "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=4, ensure_ascii=False)

    print(f"✅ 处理完成，共提取 {len(re_list)} 个名词。")
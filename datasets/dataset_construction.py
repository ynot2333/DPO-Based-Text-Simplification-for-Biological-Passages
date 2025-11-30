# import json
# from tqdm import tqdm

# datasets_path = 'datasets2'

# '''
# {
#   "id": "xxx",
#   "paragraph": "...",
#   "simplifications": [
#     {"text": "...", "score": 0.82},
#     {"text": "...", "score": 0.64},
#     {"text": "...", "score": 0.90}
#   ]
# }
# '''


# data_dict = {}
# for i in range(1, 19):
#     dict_path = f'{datasets_path}/dataset_{i}_20251122.json'
#     data_dict.update(json.load(open(dict_path, 'r')))

# pure_dict = {}
# for idx, (k, v) in tqdm(enumerate(data_dict.items()), desc='processing'):
#     name = k
#     if float(v['txt_hs_pass_rate']) >= 0.5:
#         continue
#     paragraph = v['txt_translate']
#     if len(paragraph) <= 100:
#         continue
#     simplifications = []
#     sc_in_l = []
#     for i in range(1, 6):
#         txt_n = v[f'txt{i}']
#         txt_in = txt_n['text_e']
#         if len(txt_in) <= 100:
#             continue
#         sc_in = 0.3*float(txt_n['text_e_hs_pass_rate'])+0.7*float(txt_n['text_e_completeness'])
#         sc_in_l.append(sc_in)
#         simplifications.append({'text': txt_in, 'score': sc_in})
    
#     if len(sc_in_l) < 2:
#         continue

#     max_sc_in = max(sc_in_l)
#     min_sc_in = min(sc_in_l)

#     if max_sc_in - min_sc_in < 0.1:
#         continue

#     pure_dict[idx] = {
#         'name': name,
#         'paragraph': paragraph,
#         'simplifications': simplifications
#     }

# print('total:', len(pure_dict))

# json.dump(
#     data_dict,
#     open(f'{datasets_path}/noisy_dataset2_20251122.json', 'w', encoding='utf-8'),
#     ensure_ascii=False,     # 关键：不转义中文
#     indent=4
# )
# json.dump(
#     pure_dict,
#     open('{datasets_path}/pure_dataset2_20251122.json', 'w', encoding='utf-8'),
#     ensure_ascii=False,     # 关键：不转义中文
#     indent=4
# )

import json

json1 = 'datasets/pure_dataset_20251122.json'
json2 = 'datasets2/pure_dataset2_20251122.json'

out_dict = {}
for k, v in json.load(open(json1, 'r')).items():
    out_dict[f'1_{k}'] = v

for k, v in json.load(open(json2, 'r')).items():
    out_dict[f'2_{k}'] = v

json.dump(
    out_dict,
    open('datasets_20251123.json', 'w', encoding='utf-8'),
    ensure_ascii=False,     # 关键：不转义中文
    indent=4
)

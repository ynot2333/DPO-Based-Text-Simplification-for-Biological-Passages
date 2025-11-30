import json


data_dict = {}
for i in range(1, 6):

    json_path = f'text{i}_words.json'


    data_l = json.load(open(json_path))['words_re']

    for i in data_l:
        if '/n' in i or len(i)>10 :
            continue
        data_dict[i] = data_dict.get(i,0)+1

print(len(data_dict))
high_freq_words = [w for w in data_dict if data_dict[w] >= 15]
print(high_freq_words)

json.dump(data_dict, open('texts_words.json', 'w'), ensure_ascii=False, indent=4)
with open('key_words.txt', 'w') as f:
    for word in high_freq_words:
        f.write(word + '\n')





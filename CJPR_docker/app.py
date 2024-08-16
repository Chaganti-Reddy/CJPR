import numpy as np
import torch
from keras.utils import pad_sequences
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import pandas as pd
import re
import os
import sys
import pickle
import shutil
import zipfile
from sklearn.metrics.pairwise import linear_kernel


def input_id_maker(sen, tokenizer):
    input_ids = []
    lengths = []

    sen = tokenizer.tokenize(sen, add_prefix_space=True)
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    if (len(sen) > 510):
        sen = sen[len(sen)-510:]

    sen = [CLS] + sen + [SEP]
    encoded_sent = tokenizer.convert_tokens_to_ids(sen)
    input_ids.append(encoded_sent)
    lengths.append(len(encoded_sent))

    input_ids = pad_sequences(
        input_ids, maxlen=512, value=0, dtype="long", truncating="pre", padding="post")
    return input_ids, lengths


print("Predicting")

if not os.listdir("test"):
    print("If you are running the image first time, then run the following command in the terminal & maintain the file name as year:")
    print("docker run -it --name (container_name) chagantireddy/cjpr:latest")
    print("*================================================================================================================================*")
    print(
        f"STEP:1---> docker cp file_path (container_id/container_name):/app/test")
    print(f"STEP:2---> docker start (container_id/container_name)")
    print(f"STEP:3---> docker attach (container_id/container_name)")
    print(f"Atlast save the output folder into local system using the following command:")
    print(
        f"STEP:4---> docker cp (container_id/container_name):/app/recommanded_petitions your_local_path")
    print("*================================================================================================================================*")
    print("If you already created container then run from the STEP:1 & do not create container again.\n")
    if not os.path.isdir("test"):
        os.mkdir("test")  
    sys.exit()

file = os.listdir("test")[0]
file_name = os.listdir("test")[0]
file = open("test/"+file, "r")
content = file.read()
text = content

print("Cleaning up the test folder")
[os.remove(os.path.join("test", file))
    for file in os.listdir("test")]

# model = "model.bin"
config = "config.json"
vocab = "vocab.json"
merges = "merges.txt"

tokenizer = RobertaTokenizer(vocab, merges)
# model = RobertaForSequenceClassification.from_pretrained(
    # model, config=config, ignore_mismatched_sizes=True)
model = torch.load("model.bin", map_location=torch.device('cpu'))

input_ids, lengths = input_id_maker(text, tokenizer)
input_ids = torch.tensor(input_ids)
print("*================================================================================================================================*")

with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()

    if np.argmax(logits) == 0:
        print('Petition Rejected')
    else:
        print('Petition Accepted')

print("*================================================================================================================================*")


def get_sections_used(input_string):
    pattern = r'Sections\s+([\d\s,]+)'
    pattern1 = r"Sections\s+(?P<values>\d+(?:,\s+\d+)*)\s+and"
    pattern2 = r"and\s+(?P<value>\d+)"
    pattern3 = r'Section\s+(\d+)'

    matches = re.finditer(pattern, input_string)
    matches1 = re.finditer(pattern1, input_string)
    matches2 = re.finditer(pattern2, input_string)
    matches3 = re.finditer(pattern3, input_string)

    ans_list = []

    for match in matches:
        ans_list.append(match.group(1))

    for match1 in matches1:
        ans_list.append(match1.group("values"))

    for match2 in matches2:
        ans_list.append(match2.group("value"))

    for match3 in matches3:
        ans_list.append(match3.group(1))

    ans = ", ".join(ans_list)

    dic = {}

    for i in ans_list:
        dic[i] = dic.get(i, 0) + 1

    final = []
    for key, value in dic.items():
        if value >= 1:
            final.append(key)

    return final


rows = []
if np.argmax(logits) == 0:
    print("No need to recommend similar Petitions")
    exit()

else:
    data = pd.read_csv('ILDC.csv')
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))
    data_tfidf = tfidf.transform(data['text'])
    petetion = tfidf.transform([text])

    cosine_sim = linear_kernel(petetion, data_tfidf)
    cosine_sim_dict = {i: cosine_sim[0][i] for i in range(len(cosine_sim[0]))}
    cosine_sim_dict = sorted(cosine_sim_dict.items(),
                             key=lambda x: x[1], reverse=True)

    top10 = cosine_sim_dict[:10]

    result_df = pd.DataFrame(
        columns=['s.No', 'sections_used', 'Summary_text', 'year', 'similarity_score'])

    print("Top 10 similar Petitions: ")
    for i in range(10):
        text = data.iloc[top10[i][0]]['text']
        if not os.path.exists("recommanded_petitions"):
            os.makedirs("recommanded_petitions")
        print(f'Petition{i}: {text[:100]}... with similarity {top10[i][1]}')
        with open(f'recommanded_petitions/Petition{i}.txt', 'w') as f:
            f.write(text)
        print()
        s_no = i + 1

        if get_sections_used(text):
            sections_used = get_sections_used(text)
        else:
            sections_used = "Pending Case"

        summary_text = text[:100]
        tmp = os.path.basename(file_name).split('_')[0]

        pattern = r'\b\d{4}\b'
        matches = re.findall(pattern, text)
        four_digit_numbers = list(map(int, matches))

        if len(four_digit_numbers) == 0:
            year = tmp
        else:
            for num in four_digit_numbers:
                if num >= 1947 and num < int(tmp):
                    year = num
                    break

        similarity_score = top10[i][1]
        row_dict = {'s.No': s_no, 'sections_used': sections_used,
                    'Summary_text': summary_text, 'year': year, 'similarity_score': similarity_score}
        rows.append(row_dict)

    result_df = pd.DataFrame(rows)
    result_df.to_csv('recommanded_petitions/result_table.csv', index=False)
    print(result_df)
    print("Top 10 similar Petitions & Results saved in 'recommanded_petitions' folder")
    print
    ("*================================================================================================================================*")

import json
import os
import re

import nltk.data
import pandas as pd
import progressbar
from nltk.tokenize import word_tokenize

nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# give path to the directory where all the case files are contained
input_paths_of_files = "Preprocessed_files/"
files = os.listdir(input_paths_of_files)

data = {}
data['data'] = {}


# an implementation to find the label of each case, assumption that every case has words like accepted,rejected etc. at the end of the case
def label_for_file(sentence_set):
    label = -1
    sentnum = -1
    # loop through all sentences in which we will check for the words like accepted, rejected etc.
    for i in range(len(sentence_set)):
        sentence = sentence_set[i]
        sentence = sentence.lower()
        # for accepted cases we check words like accepted, approved, permitted and their variations
        A_accepted = ("accepted" in sentence) or (
                "accept" in sentence) or ("accepting" in sentence)
        A_allowed = ("allowed" in sentence) or (
                "allow" in sentence) or ("allowing" in sentence)
        A_permitted = ("permitted" in sentence) or (
                "permit" in sentence) or ("permitting" in sentence)
        A = A_accepted or A_allowed or A_permitted
        # for rejected cases we check words like rejected, disposed, dismissed and their variations
        R_rejected = ("rejected" in sentence) or (
                "reject" in sentence) or ("rejecting" in sentence)
        R_disposed = ("disposed" in sentence) or (
                "dispose" in sentence) or ("disposing" in sentence)
        R_dismissed = ("dismissed" in sentence) or (
                "dismiss" in sentence) or ("dismissing" in sentence)
        R = R_rejected or R_dismissed or R_disposed
        # we check if any sentence has accepted type and appeal together in a sentence and similarly for rejected type
        Appeal = ("appeal" in sentence) or ("appeals" in sentence)
        App = Appeal
        Not = ("not" in sentence) or ("no" in sentence)
        if App and A and R and Not:
            break
        if App and A and R:
            break
        if App and A:
            label = 1
            sentnum = i
            break
        if App and R:
            label = 0
            sentnum = i
            break
        if App and A and Not:
            break
        if App and R and Not:
            break

    return label, sentnum


civil_cases_pattern = ['civil']
criminal_cases_pattern = ['criminal', 'Cr.P.C', 'Cr PC']

like_orig_path = "A/"
summ_feed = "B/"

for i in progressbar.progressbar(range(len(files))):
    file_path = "Preprocessed_files/" + files[i]
    f = open(file_path, "r")
    text = f.read()

    sentences = tokenizer.tokenize(text)
    # we consider last 100 sentences where we check for label
    sentence_set = sentences[max(0, len(sentences) - 100):]
    sentence_set.reverse()  # reverse the set of sentences

    label, sentnum = label_for_file(sentence_set)

    if label == -1:  # if no label is found skip case
        continue

    # after label is found, cut the end part of text where label was found
    cut_sentences = sentences[:len(sentences) - sentnum - 1]
    new_sentence_set = sentences[max(0, len(cut_sentences) - 10):]
    new_sentence_set.reverse()

    new_label, new_sentnum = label_for_file(new_sentence_set)

    if new_label != label:
        new_text = " ".join(cut_sentences)
        final_text = new_text

        if label == 1:  # label 1 for accepted
            if not os.path.exists(like_orig_path):
                os.mkdir(like_orig_path)
            p_file = open(like_orig_path + "Accepted/" + files[i], "w")
            p_file.write(final_text)  # write the final text in a file
            tokens = word_tokenize(final_text)
            num_tokens = len(tokens)
            data['data'][files[i]] = {}
            # the name key contains the name of file
            data['data'][files[i]]['name'] = files[i]
            # the text key contains the text of file
            data['data'][files[i]]['text'] = final_text
            # the year key contains the year of case
            data['data'][files[i]]['year'] = files[i][:4]
            # the sentences key contains the length of all sentences after removing the part which helped in label finding
            data['data'][files[i]]['sentences'] = len(cut_sentences)
            # the tag key contains the tag of case
            for pattern in civil_cases_pattern:
                if re.search(pattern, text, re.IGNORECASE):
                    data['data'][files[i]]['tag'] = 'Civil'
                    break
            for pattern in criminal_cases_pattern:
                if re.search(pattern, text, re.IGNORECASE):
                    data['data'][files[i]]['tag'] = 'Criminal'
                    break
            # the label key contains the label of case
            data['data'][files[i]]['label'] = 'Accepted'
            p_file.close()
            lines = final_text.split("\n")
            text_new = " ".join(lines)
            text_new = re.sub(" +", " ", text_new)
            if not os.path.exists(summ_feed):
                os.mkdir(summ_feed)
            CT_file = open(summ_feed + "Accepted/" + files[i], "w")
            CT_file.write(text_new)
            CT_file.close()

        elif label == 0:  # label 0 for rejected
            if not os.path.exists('Rejected'):
                os.mkdir('Rejected')
            p_file = open(like_orig_path + "Rejected/" + files[i], "w")
            p_file.write(final_text)
            tokens = word_tokenize(final_text)
            num_tokens = len(tokens)
            data['data'][files[i]] = {}
            data['data'][files[i]]['name'] = files[i]
            data['data'][files[i]]['text'] = final_text
            data['data'][files[i]]['year'] = files[i][:4]
            data['data'][files[i]]['sentences'] = len(cut_sentences)
            for pattern in civil_cases_pattern:
                if re.search(pattern, text, re.IGNORECASE):
                    data['data'][files[i]]['tag'] = 'Civil'
                    break
            for pattern in criminal_cases_pattern:
                if re.search(pattern, text, re.IGNORECASE):
                    data['data'][files[i]]['tag'] = 'Criminal'
                    break
            data['data'][files[i]]['label'] = 'Rejected'
            p_file.close()
            lines = final_text.split("\n")
            text_new = " ".join(lines)
            text_new = re.sub(" +", " ", text_new)
            CT_file = open(summ_feed + "Rejected/" + files[i], "w")
            CT_file.write(text_new)
            CT_file.close()

    else:
        new_cut_sentences = cut_sentences[:len(cut_sentences) - new_sentnum - 1]
        new_text = " ".join(new_cut_sentences)
        final_text = new_text

        if label == 1:
            p_file = open(like_orig_path + "Accepted/" + files[i], "w")
            p_file.write(final_text)
            tokens = word_tokenize(final_text)
            num_tokens = len(tokens)
            data['data'][files[i]] = {}
            data['data'][files[i]]['name'] = files[i]
            data['data'][files[i]]['text'] = final_text
            data['data'][files[i]]['year'] = files[i][:4]
            data['data'][files[i]]['sentences'] = len(new_cut_sentences)
            for pattern in civil_cases_pattern:
                if re.search(pattern, text, re.IGNORECASE):
                    data['data'][files[i]]['tag'] = 'Civil'
                    break
            for pattern in criminal_cases_pattern:
                if re.search(pattern, text, re.IGNORECASE):
                    data['data'][files[i]]['tag'] = 'Criminal'
                    break
            data['data'][files[i]]['label'] = 'Accepted'

            p_file.close()
            lines = final_text.split("\n")
            text_new = " ".join(lines)
            text_new = re.sub(" +", " ", text_new)
            CT_file = open(summ_feed + "Accepted/" + files[i], "w")
            CT_file.write(text_new)
            CT_file.close()

        elif label == 0:
            p_file = open(like_orig_path + "Rejected/" + files[i], "w")
            p_file.write(final_text)
            tokens = word_tokenize(final_text)
            num_tokens = len(tokens)
            data['data'][files[i]] = {}
            data['data'][files[i]]['name'] = files[i]
            data['data'][files[i]]['text'] = final_text
            data['data'][files[i]]['sentences'] = len(new_cut_sentences)
            data['data'][files[i]]['year'] = files[i][:4]
            for pattern in civil_cases_pattern:
                if re.search(pattern, text, re.IGNORECASE):
                    data['data'][files[i]]['tag'] = 'Civil'
                    break
            for pattern in criminal_cases_pattern:
                if re.search(pattern, text, re.IGNORECASE):
                    data['data'][files[i]]['tag'] = 'Criminal'
                    break
            data['data'][files[i]]['label'] = 'Rejected'
            p_file.close()
            lines = final_text.split("\n")
            text_new = " ".join(lines)
            text_new = re.sub(" +", " ", text_new)
            CT_file = open(summ_feed + "Rejected/" + files[i], "w")
            CT_file.write(text_new)
            CT_file.close()

# now save the 'data' dictionary  in a json file
json_file = open("splitter.json", "w")
json.dump(data, json_file)

# make csv
df = pd.DataFrame(data['data'])
df = df.transpose()
df.to_csv('data_1.csv', index=False)

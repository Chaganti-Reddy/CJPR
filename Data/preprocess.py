import os
import re

import nltk.data
import progressbar
from nltk.tokenize import word_tokenize

nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

path_dataset_dir = "Yearwise_data/"  # give the path of directory where the dataset is saved
files = os.listdir(path_dataset_dir)  # all the files in the dataset

# counter to count how many files have been preprocessed at a certain instance
num_files_till_now = 0

for j in progressbar.progressbar(range(len(files))):
    sub_dir = os.listdir(path_dataset_dir + files[j])
    for k in progressbar.progressbar(range(len(sub_dir))):
        file_path = os.path.join(path_dataset_dir, files[j], sub_dir[k])
        f = open(file_path, "r",encoding="utf8")
        text = f.read()
        text = re.sub(r"\xa0", " ", text)
        text = text.split("\n")  # splitting using new line character
        # removing everything other than these a-zA-Z0-9.,)\-(/?\t
        text = [re.sub(r'[^a-zA-Z0-9.,)\-(/?\t ]', '', sentence)
                for sentence in text]
        text = [re.sub(r'(?<=[^0-9])/(?=[^0-9])', ' ', sentence)
                for sentence in text]
        # converting multiple tabs and spaces ito a single tab or space
        text = [re.sub("\t+", " ", sentence) for sentence in text]
        text = [re.sub(" +", " ", sentence) for sentence in text]
        # these were the common noises in out data, depends on data
        text = [re.sub("\.\.+", "", sentence) for sentence in text]
        text = [re.sub("\A ?", "", sentence) for sentence in text]
        text = [sentence for sentence in text if (
                len(sentence) != 1 and not re.fullmatch("(\d|\d\d|\d\d\d)", sentence))]
        text = [re.sub('\A\(?(\d|\d\d\d|\d\d|[a-zA-Z])(\.|\))\s?(?=[A-Z])', '\n', sentence)
                for sentence in text]  # dividing into para wrt to points
        text = [re.sub("\A\(([ivx]+)\)\s?(?=[a-zA-Z0-9])", '\n', sentence)
                for sentence in text]  # dividing into para wrt to roman points
        text = [re.sub(r"[()[\]\"$']", " ", sentence)
                for sentence in text]  # removing ()[\]\"$' these characters
        # converting no., nos., co., ltd.  to number, numbers, company and limited
        text = [re.sub(r" no.", " number", sentence) for sentence in text]
        text = [re.sub(r" nos.", " numbers", sentence) for sentence in text]
        text = [re.sub(r" co.", " company", sentence) for sentence in text]
        text = [re.sub(r" ltd.", " limited", sentence) for sentence in text]
        text2 = []
        for index in range(len(text)):  # for removing multiple new-lines
            if index > 0 and text[index] == '' and text[index - 1] == '':
                continue
            if index < len(text) - 1 and text[index + 1] != '' and text[index + 1][0] == '\n' and text[index] == '':
                continue
            text2.append(text[index])
        text = text2
        for i in range(len(text)):  # ignoring the text before JUDGMENT,ORDER......
            if re.search("\A(ORDER|JUDGMENT|J U D G M E N T|O R D E R)", text[i]):
                break
            # if (i == len(text)-1):
            #     continue
            # if (re.search("\A(ORDER|JUDGMENT|J U D G M E N T|O R D E R)", text[i+1])):
            #     i = i+1
        text = text[i + 1:]
        text = "\n".join(text)
        lines = text.split("\n")
        text_new = " ".join(lines)  # joining all the lines into a single text
        text_new = re.sub(" +", " ", text_new)
        num_tokens = len(word_tokenize(text_new))
        num_files_till_now += 1
        if num_files_till_now % 1000 == 0:
            print("\nNumber of files that have been preprocessed: {0}".format(
                num_files_till_now))
        if num_tokens < 100:  # if number of tokens in file after preprocessing is less than 100 skipping the file
            continue
        # finally writing the preprocessed file in output directory
        if not os.path.exists('Preprocessed_files'):
            os.mkdir('Preprocessed_files')
        fff = open("Preprocessed_files/" + sub_dir[k] + ".txt", "w+")
        fff.write(text_new)
        fff.close()
    print(f"Year {files[j]} done")

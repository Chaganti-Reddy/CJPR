import os

a_directory = "A"
preprocessed_directory = "Preprocessed_files"

a_files = set(os.listdir(os.path.join(a_directory, "Accepted")) + os.listdir(os.path.join(a_directory, "Rejected")))
preprocessed_files = set(os.listdir(preprocessed_directory))

Pre_count = len(preprocessed_files)
a_count = len(a_files)

unknown_files = sorted(preprocessed_files - a_files)

count = 0

with open("unknown.txt", "w") as f:
    for filename in unknown_files:
        f.write(filename + "\n")
        count += 1

print("Unknown files have been written to unknown.txt")
print(f"Total No.of Unknown Files Saved are: {count} and No.of Preprocessed_Files - Labeled_Data = {Pre_count - a_count}")

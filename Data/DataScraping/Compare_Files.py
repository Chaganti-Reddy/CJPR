import filecmp
import os


def compare_and_delete(folder1, folder2):
    dcmp = filecmp.dircmp(folder1, folder2)

    # Compare files in common between the two folders
    for common_file in dcmp.common_files:
        file1 = os.path.join(folder1, common_file)
        file2 = os.path.join(folder2, common_file)

        # Check if files are identical
        if filecmp.cmp(file1, file2, shallow=False):
            print(
                f"Files {file1} and {file2} are identical. Deleting {file1}.")
            os.remove(file1)


# Example usage
folder_path1 = 'C:\\Users\\ramch\\Downloads\\2023'
folder_path2 = 'r:\\CJPR\\Data\\DataScraping\\Yearwise_data\\2023'

compare_and_delete(folder_path1, folder_path2)

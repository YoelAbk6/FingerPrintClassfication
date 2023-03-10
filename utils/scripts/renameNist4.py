import os

# Define a function to rename a single file based on the gender information in its corresponding TXT file


def rename_file(folder_path, file_name):
    txt_file_name = os.path.splitext(file_name)[0] + ".txt"
    txt_file_path = os.path.join(folder_path, txt_file_name)
    with open(txt_file_path, "r") as f:
        lines = f.readlines()
        gender = lines[0].strip().split(": ")[1]
    new_file_name = file_name[:5] + "_" + gender + "_" + file_name[6:]
    os.rename(os.path.join(folder_path, file_name),
              os.path.join(folder_path, new_file_name))

# Define a function to iterate through all the files in a folder and call rename_file() on each one


def rename_files_in_folder(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".png"):
            rename_file(folder_path, file_name)


# Iterate through all the folders and call rename_files_in_folder() on each one
for i in range(8):
    folder_path = f"figs_{i}"
    rename_files_in_folder(folder_path)

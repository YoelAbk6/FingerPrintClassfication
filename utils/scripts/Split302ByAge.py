import os
import glob
import shutil

folder_path = ""

# Create the output directories if they don't already exist
age_group_dirs = ["18to38", "38to58"]
age_dirs = ["18to28", "28to38", "38to48", "48to58"]
for age_group_dir in age_group_dirs:
    os.makedirs(os.path.join(folder_path, age_group_dir), exist_ok=True)
    for age_dir in age_dirs:
        os.makedirs(os.path.join(folder_path, age_dir), exist_ok=True)

# Copy the images to the appropriate folders
for filename in glob.glob(os.path.join(folder_path, "*.png")):
    name, ext = os.path.splitext(os.path.basename(filename))
    age = int(name.split("_")[-1])

    # Copy to age group directories
    if 18 <= age < 38:
        shutil.copy2(filename, os.path.join(folder_path, "18to38"))
    elif 38 <= age <= 58:
        shutil.copy2(filename, os.path.join(folder_path, "38to58"))

    # Copy to age directories
    if 18 <= age <= 28:
        shutil.copy2(filename, os.path.join(folder_path, "18to28"))
    elif 28 < age <= 38:
        shutil.copy2(filename, os.path.join(folder_path, "28to38"))
    elif 38 < age <= 48:
        shutil.copy2(filename, os.path.join(folder_path, "38to48"))
    elif 48 < age <= 58:
        shutil.copy2(filename, os.path.join(folder_path, "48to58"))

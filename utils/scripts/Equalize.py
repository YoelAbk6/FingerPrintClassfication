import os
import shutil

path = 'D:/data/NIST302/images/auxiliary/flat/M/500/plain/png/regular'
# path = 'D:/data/SOCOFing/Real'
males = []
females = []

# Iterate over all files in the directory
for filename in os.listdir(path):
    # Check if file has the specified name format
    if filename.endswith('.png'):  # or filename.endswith('.BMP'):
        name_parts = filename.split("_")
        if name_parts[-2] == "M":
            males.append(filename)
        elif name_parts[-2] == "F":
            females.append(filename)

# Determine the maximum number of unique images from each gender
max_images = min(len(males), len(females))

# Copy the maximum number of unique images from each gender to the same new folder
male_dest_folder = 'D:/data/NIST302/images/auxiliary/flat/M/500/plain/png/equal/males'
female_dest_folder = 'D:/data/NIST302/images/auxiliary/flat/M/500/plain/png/equal/females'
# dest_folder = 'D:/data/SOCOFing/Real/equal'

if not os.path.exists(male_dest_folder):
    os.makedirs(male_dest_folder)
if not os.path.exists(female_dest_folder):
    os.makedirs(female_dest_folder)

for i in range(max_images):
    shutil.copy2(path + '/' + males[i], male_dest_folder)
    shutil.copy2(path + '/' + females[i], female_dest_folder)

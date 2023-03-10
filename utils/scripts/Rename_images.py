import os
import pandas as pd

# Path to the folder containing the images
image_folder_path = "D:/data/NIST302/images/auxiliary/flat/*/500/plain/png"

# Path to the CSV file
csv_file_path = "D:/NIST302/participants.csv"

# Load the CSV file into a pandas dataframe
df = pd.read_csv(csv_file_path)

for letter in ['U', 'V']:
    current_image_folder_path = image_folder_path.replace('*', letter)

    # Iterate through all the images in the image folder
    for image_name in os.listdir(current_image_folder_path):
        # Extract the subject id from the image name
        subject_id = image_name[:8].lstrip("0")
        # Find the corresponding row in the dataframe for the subject id
        subject_row = df[df['id'] == int(subject_id)]
        # Extract the gender from the dataframe
        age = subject_row['age'].values[0]
        # Add the age to the image name
        new_image_name = image_name[:-4] + f"_{age}.png"
        # Rename the image
        os.rename(os.path.join(current_image_folder_path, image_name),
                  os.path.join(current_image_folder_path, new_image_name))

    print(f"Finished renaming images {letter}")

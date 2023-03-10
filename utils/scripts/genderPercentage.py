import os

male_count = 0
female_count = 0

# Iterate through all the folders and all the image files to count the number of males and females
for i in range(8):
    folder_path = f"figs_{i}"
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".png"):
            gender = file_name.split("_")[1]
            if gender == "M":
                male_count += 1
            elif gender == "F":
                female_count += 1

# Compute the male and female percentages
total_count = male_count + female_count
male_percent = (male_count / total_count) * 100
female_percent = (female_count / total_count) * 100

print(f"Total: {total_count}")
print(f"Male: {male_count} ({male_percent:.2f}%)")
print(f"Female: {female_count} ({female_percent:.2f}%)")

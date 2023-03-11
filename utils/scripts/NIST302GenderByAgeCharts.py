import os
import glob
import matplotlib.pyplot as plt

folder_path = "D:/data/NIST302/images/auxiliary/flat/M/500/plain/png/regular"
age_counts = {}
gender_counts = {"F": {"18 to 38": 0, "38 to 58": 0},
                 "M": {"18 to 38": 0, "38 to 58": 0}}

# Count the number of images in each age and gender group
for filename in glob.glob(os.path.join(folder_path, "*.png")):
    name, ext = os.path.splitext(os.path.basename(filename))
    age = int(name.split("_")[-1])
    gender = name.split("_")[-2]

    if 18 <= age < 38:
        age_group = "18 to 38"
    elif 38 <= age <= 58:
        age_group = "38 to 58"
    else:
        continue

    if gender not in gender_counts:
        continue

    if age_group not in age_counts:
        age_counts[age_group] = 0
    age_counts[age_group] += 1

    gender_counts[gender][age_group] += 1

# Create pie charts for the gender split in each age group
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

for i, (age_group, count) in enumerate(age_counts.items()):
    ax = axs[1-i]  # Reverse the order of the axes
    ax.set_title(f"Age group: {age_group}")
    labels = ["Female", "Male"]
    sizes = [gender_counts["F"][age_group], gender_counts["M"][age_group]]
    ax.pie(sizes, labels=labels, autopct='%1.1f%%',
           startangle=90, colors=["hotpink", "steelblue"])
    ax.axis('equal')
    ax.set_xlabel(f"Total images: {count}")

plt.show()

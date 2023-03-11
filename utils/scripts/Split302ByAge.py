import os
import shutil

# define the source and destination paths
src_path = 'data/NIST302/auxiliary/flat/M/500/plain/png/regular'
dst_path_2_split = 'data/NIST302/auxiliary/flat/M/500/plain/png/age-2-split'
dst_path_4_split = 'data/NIST302/auxiliary/flat/M/500/plain/png/age-4-split'

# define the age ranges
age_ranges_2_split = {
    '18to38': (18, 38),
    '38to58': (38, 58)
}

age_ranges_4_split = {
    '18to28': (18, 28),
    '28to38': (28, 38),
    '38to48': (38, 48),
    '48to58': (48, 58)
}

# create the destination folders
for dst_dir in [dst_path_2_split, dst_path_4_split]:
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

# copy the images to the 2-split folders
for filename in os.listdir(src_path):
    age = int(filename.split('_')[-1].replace('.png', ''))
    for dst_dir, age_range in age_ranges_2_split.items():
        if age >= age_range[0] and age < age_range[1]:
            dst_path = os.path.join(dst_path_2_split, dst_dir)
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            shutil.copy(os.path.join(src_path, filename), dst_path)

# copy the images to the 4-split folders
for filename in os.listdir(src_path):
    age = int(filename.split('_')[-1].replace('.png', ''))
    for dst_dir, age_range in age_ranges_4_split.items():
        if age >= age_range[0] and age <= age_range[1]:
            dst_path = os.path.join(dst_path_4_split, dst_dir)
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            shutil.copy(os.path.join(src_path, filename), dst_path)

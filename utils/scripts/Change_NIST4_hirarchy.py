import os
import shutil

def move_f_images_to_dest(src, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for file_name in os.listdir(src):
        if file_name.endswith(".png") and file_name.startswith('f'):
            shutil.copy2(src + '/' + file_name, dst)

    


def change_nist4_hir():
    dst_path = f"data/sd04/png_txt/figs"
    for i in range(8):
        src_path = f"data/sd04/png_txt/figs_{i}"
        move_f_images_to_dest(src_path,dst_path)

if __name__ == '__main__':
    change_nist4_hir()

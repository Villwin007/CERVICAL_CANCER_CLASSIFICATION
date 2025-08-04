import os
import shutil

base_dir = "data/raw/cervicalCancer"

for class_folder in os.listdir(base_dir):
    outer_path = os.path.join(base_dir, class_folder)
    inner_path = os.path.join(outer_path, class_folder)

    # Check if the nested folder exists
    if os.path.isdir(inner_path):
        for fname in os.listdir(inner_path):
            src = os.path.join(inner_path, fname)
            dst = os.path.join(outer_path, fname)

            # Move only .bmp files
            if fname.lower().endswith(".bmp") and os.path.isfile(src):
                shutil.move(src, dst)
                print(f"Moved: {fname}")

        # Check if inner folder is empty before removing
        if len(os.listdir(inner_path)) == 0:
            os.rmdir(inner_path)
            print(f"✅ Removed empty folder: {inner_path}")
        else:
            print(f"⚠️ Skipped deletion (folder not empty): {inner_path}")
    else:
        print(f"✅ Already flat: {outer_path}")

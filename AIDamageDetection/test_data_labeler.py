import os
import shutil
import pandas as pd

# folder dataset kamu
base_dir = "data_test"

# output folder
output_dir = "data_test_labeled"
os.makedirs(output_dir, exist_ok=True)

labels = []

# semua class = nama folder
classes = os.listdir(base_dir)

for cls in classes:
    class_path = os.path.join(base_dir, cls)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)

    count = 1

    for img in images:
        old_path = os.path.join(class_path, img)

        if not os.path.isfile(old_path):
            continue

        # ambil extension
        ext = os.path.splitext(img)[1]

        # rename format
        new_name = f"{cls}_{count}{ext}"
        new_path = os.path.join(output_dir, new_name)

        # copy + rename
        shutil.copy2(old_path, new_path)

        # simpan label
        labels.append([new_name, cls])

        count += 1

# simpan CSV label
df = pd.DataFrame(labels, columns=["filename", "label"])
df.to_csv("labels.csv", index=False)

print("DONE ✔")
print("Images saved to:", output_dir)
print("Labels saved to: labels.csv")
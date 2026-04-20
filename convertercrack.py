import os
import cv2

# path
image_dir = "dataset/images"
mask_dir = "dataset/masks"
label_dir = "dataset/labels"

os.makedirs(label_dir, exist_ok=True)

for filename in os.listdir(mask_dir):
    if not filename.endswith(".png"):
        continue

    mask_path = os.path.join(mask_dir, filename)
    image_path = os.path.join(image_dir, filename.replace(".png", ".jpg"))

    # baca mask (grayscale)
    mask = cv2.imread(mask_path, 0)
    if mask is None:
        continue

    h, w = mask.shape

    # threshold biar jelas putihnya
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # cari contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    label_path = os.path.join(label_dir, filename.replace(".png", ".txt"))

    with open(label_path, "w") as f:
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)

            # filter kecil (noise)
            if bw * bh < 100:
                continue

            # convert ke YOLO format
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            bw /= w
            bh /= h

            class_id = 0  # crack

            f.write(f"{class_id} {x_center} {y_center} {bw} {bh}\n")
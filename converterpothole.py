import xml.etree.ElementTree as ET
import os

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x*dw, y*dh, w*dw, h*dh)

classes = ["pothole"]

xml_folder = "annotations"
yolo_folder = "labels"

os.makedirs(yolo_folder, exist_ok=True)

for file in os.listdir(xml_folder):
    tree = ET.parse(os.path.join(xml_folder, file))
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    txt_file = open(os.path.join(yolo_folder, file.replace(".xml", ".txt")), "w")

    for obj in root.iter("object"):
        cls = obj.find("name").text
        if cls not in classes:
            continue

        xmlbox = obj.find("bndbox")
        b = (
            float(xmlbox.find("xmin").text),
            float(xmlbox.find("xmax").text),
            float(xmlbox.find("ymin").text),
            float(xmlbox.find("ymax").text),
        )

        bb = convert((w, h), b)
        txt_file.write(f"0 {' '.join(map(str, bb))}\n")

    txt_file.close()
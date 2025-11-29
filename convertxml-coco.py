import os
import json
import xml.etree.ElementTree as ET

# ======= 需要你修改的三处路径 =======

ANNOT_DIR = r"G:\域适应目标检测识别数据集\舰船公开样本数据集\LS-SSDD\LS-SSDD-v1.0-OPEN\Annotations"
# 图像所在文件夹（你自己确认一下名称，比如 JPEGImages 或 images）
IMAGE_DIR = r"G:\域适应目标检测识别数据集\舰船公开样本数据集\LS-SSDD\LS-SSDD-v1.0-OPEN\JPEGImages_sub"
# 输出的 COCO 标注文件（建议放在某个 annotations 目录下）
OUTPUT_JSON = r"G:\域适应目标检测识别数据集\舰船公开样本数据集\LS-SSDD\LS-SSDD-v1.0-OPEN\annotations\test.json"

# 如果输出目录不存在，先建一下
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

# ======= COCO 格式初始化 =======
coco = {
    "images": [],
    "annotations": [],
    "categories": [
        {
            "id": 1,
            "name": "ship",
            "supercategory": "ship"
        }
    ]
}

image_id = 1
ann_id = 1

xml_files = [f for f in os.listdir(ANNOT_DIR) if f.endswith(".xml")]
xml_files.sort()

for xml_name in xml_files:
    xml_path = os.path.join(ANNOT_DIR, xml_name)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 读取文件名
    filename = root.findtext("filename")
    if filename is None:
        # 有的 VOC 标注里文件名在 <name> 或其他字段，可以根据实际 xml 内容调整
        filename = xml_name.replace(".xml", ".jpg")

    # 读取图像宽高（来自 XML 的 <size>）
    size = root.find("size")
    if size is None:
        print(f"Warning: no <size> tag in {xml_name}, skip.")
        continue

    width = int(size.findtext("width"))
    height = int(size.findtext("height"))

    # 检查图片文件是否存在
    img_path = os.path.join(IMAGE_DIR, filename)
    if not os.path.exists(img_path):
        # 有的标注可能是 .jpg / .png 混着，可以做一点兼容
        alt_jpg = filename.rsplit(".", 1)[0] + ".jpg"
        alt_png = filename.rsplit(".", 1)[0] + ".png"
        if os.path.exists(os.path.join(IMAGE_DIR, alt_jpg)):
            filename = alt_jpg
            img_path = os.path.join(IMAGE_DIR, filename)
        elif os.path.exists(os.path.join(IMAGE_DIR, alt_png)):
            filename = alt_png
            img_path = os.path.join(IMAGE_DIR, filename)
        else:
            print(f"Warning: image file for {xml_name} not found, skip.")
            continue

    # 添加 image 信息
    coco["images"].append({
        "id": image_id,
        "file_name": filename,
        "width": width,
        "height": height
    })

    # 遍历每个目标
    for obj in root.findall("object"):
        cls_name = obj.findtext("name")
        if cls_name is None:
            continue

        # SSDD 里通常就是 ship，一律归为 ship 类
        cls_name = cls_name.strip().lower()
        if "ship" not in cls_name:
            # 如果有其他类别，你可以根据需要调整；这里简单忽略非 ship
            continue

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = float(bndbox.findtext("xmin"))
        ymin = float(bndbox.findtext("ymin"))
        xmax = float(bndbox.findtext("xmax"))
        ymax = float(bndbox.findtext("ymax"))

        # 转成 COCO 格式的 [x, y, w, h]
        x = xmin
        y = ymin
        w = xmax - xmin
        h = ymax - ymin

        if w <= 0 or h <= 0:
            continue

        ann = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": 1,  # ship
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0
        }
        coco["annotations"].append(ann)
        ann_id += 1

    image_id += 1

print(f"Parsed {len(coco['images'])} images, {len(coco['annotations'])} objects.")

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(coco, f, ensure_ascii=False, indent=2)

print(f"Saved COCO annotations to {OUTPUT_JSON}")

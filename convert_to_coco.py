import os
import json
from PIL import Image
from datetime import datetime
import glob

def create_coco_structure():
    return {
        "info": {
            "year": str(datetime.now().year),
            "version": "1",
            "description": "Converted from YOLO format",
            "contributor": "Dataset Converter",
            "date_created": datetime.now().strftime("%Y-%m-%dT%H:%M:%S+00:00")
        },
        "licenses": [
            {
                "id": 1,
                "url": "https://creativecommons.org/publicdomain/zero/1.0/",
                "name": "Public Domain"
            }
        ],
        "categories": [
            {
                "id": 0,
                "name": "player",
                "supercategory": "none"
            },
            {
                "id": 1,
                "name": "ball",
                "supercategory": "none"
            },
            {
                "id": 2,
                "name": "logo",
                "supercategory": "none"
            }
        ],
        "images": [],
        "annotations": []
    }

def convert_yolo_to_coco(yolo_box, img_width, img_height):
    # YOLO format: [x_center, y_center, width, height] (normalized)
    x_center, y_center, width, height = yolo_box
    
    # Convert to absolute coordinates
    x = x_center * img_width
    y = y_center * img_height
    w = width * img_width
    h = height * img_height
    
    # Convert to COCO format [x, y, width, height]
    # COCO uses top-left corner coordinates
    x = x - w/2
    y = y - h/2
    
    return [x, y, w, h]

def process_dataset(base_path, split):
    coco_data = create_coco_structure()
    annotation_id = 0
    
    # Get all image files
    image_files = glob.glob(os.path.join(base_path, split, "images", "*"))
    
    for img_idx, img_path in enumerate(image_files):
        # Get image dimensions
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        # Create image entry
        image_entry = {
            "id": img_idx,
            "license": 1,
            "file_name": os.path.basename(img_path),
            "height": img_height,
            "width": img_width,
            "date_captured": datetime.now().strftime("%Y-%m-%dT%H:%M:%S+00:00")
        }
        coco_data["images"].append(image_entry)
        
        # Get corresponding label file
        label_path = os.path.join(base_path, split, "labels", 
                                 os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    
                    # Convert YOLO format to COCO format
                    bbox = convert_yolo_to_coco([x_center, y_center, width, height], 
                                              img_width, img_height)
                    
                    # Calculate area
                    area = bbox[2] * bbox[3]
                    
                    # Create annotation entry
                    annotation_entry = {
                        "id": annotation_id,
                        "image_id": img_idx,
                        "category_id": int(class_id),
                        "bbox": bbox,
                        "area": area,
                        "segmentation": [],
                        "iscrowd": 0
                    }
                    coco_data["annotations"].append(annotation_entry)
                    annotation_id += 1
    
    return coco_data

def main():
    base_path = "dataset/1561_SmartCrop"
    
    # Process each split
    for split in ["train", "valid", "test"]:
        print(f"Processing {split} split...")
        coco_data = process_dataset(base_path, split)
        
        # Save to JSON file
        output_file = os.path.join(base_path, f"{split}_coco.json")
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=4)
        print(f"Saved {split} annotations to {output_file}")

if __name__ == "__main__":
    main() 
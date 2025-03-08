import os
import json
from PIL import Image

# Function to load .odgt file
def load_odgt(odgt_path):
    data = []
    with open(odgt_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

# Function to convert bounding box to YOLO format
def convert_to_yolo_format(bbox, image_width, image_height):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2 / image_width
    y_center = (y_min + y_max) / 2 / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    return x_center, y_center, width, height

# Function to extract person bounding boxes
def extract_person_boxes(data, images_dir, labels_dir):
    for entry in data:
        image_id = entry['ID']
        image_path = os.path.join(images_dir, f"{image_id}.jpg")  # Assuming images are .jpg
        if not os.path.exists(image_path):
            print(f"Image {image_id}.jpg not found. Skipping...")
            continue

        # Get image dimensions
        with Image.open(image_path) as img:
            image_width, image_height = img.size

        # Prepare YOLO label content
        yolo_lines = []
        for box in entry['gtboxes']:
            if box['tag'] == 'person':  # Only consider 'person' class
                # Use 'fbox' for full body bounding box
                x_min, y_min, x_max, y_max = box['fbox']
                x_center, y_center, width, height = convert_to_yolo_format(
                    [x_min, y_min, x_max, y_max], image_width, image_height
                )
                yolo_lines.append(f"0 {x_center} {y_center} {width} {height}")  # Class ID for person is 0

        # Save YOLO label file
        yolo_label_path = os.path.join(labels_dir, f"{image_id}.txt")
        with open(yolo_label_path, 'w') as file:
            file.write("\n".join(yolo_lines))

        print(f"Created YOLO label file: {yolo_label_path}")

# Main function
def main():
    # Path to .odgt file
    odgt_path = "/Users/himanshu-r/Documents/Project/human-movement-prediction/test_code/CrowdHuman/annotation_val.odgt"  # Path to CrowdHuman annotation file

    # Path to images directory
    images_dir = "/Users/himanshu-r/Documents/Project/human-movement-prediction/test_code/CrowdHuman/images/val"  # Path to training images

    # Path to save YOLO label files
    labels_dir = "/Users/himanshu-r/Documents/Project/human-movement-prediction/test_code/CrowdHuman/labels/val"  # Directory to save YOLO label files

    # Create labels directory if it doesn't exist
    os.makedirs(labels_dir, exist_ok=True)

    # Load .odgt file
    data = load_odgt(odgt_path)

    # Extract person bounding boxes and save as YOLO labels
    extract_person_boxes(data, images_dir, labels_dir)

if __name__ == "__main__":
    main()
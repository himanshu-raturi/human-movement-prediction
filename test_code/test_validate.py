from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO("yolov8l.pt")  # You can use yolov8s, yolov8m, yolov8l, or yolov8x

# Train the model
def train():
    results = model.train(
        data="/Users/himanshu-r/Documents/Project/human-movement-prediction/test_code/data.yaml",  # Path to data.yaml
        epochs=50,                            # Number of training epochs
        batch=4,                             # Batch size
        imgsz=640,                            # Image size
        device="cpu",                         # Use CPU (or "mps" for Apple Silicon M1/M2)
        name="yolov8_crowdhuman"              # Name of the training run
    )
# Evaluate the model
metrics = model.val()  # Evaluate on the validation set
print(metrics.box.map)  # Print mAP score
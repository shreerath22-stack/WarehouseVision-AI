from ultralytics import YOLO
from pathlib import Path


# --------------------------------------------------
# 1. Load the YOLO model
# --------------------------------------------------

print("Loading YOLO model...")

model = YOLO("yolo11n.pt")

print("YOLO model loaded successfully!")


# --------------------------------------------------
# 2. Run object detection
# --------------------------------------------------

print("Running object detection...")

results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    conf=0.25,
    save=True,
    project="outputs",
    name="yolo_test",
    exist_ok=True
)


# --------------------------------------------------
# 3. Display detected objects
# --------------------------------------------------

print("\nDetection completed!")

for result in results:
    print("\nDetected objects:")

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        object_name = model.names[class_id]

        print(
            f"- {object_name}: "
            f"{confidence:.2%} confidence"
        )


# --------------------------------------------------
# 4. Show output location
# --------------------------------------------------

output_path = Path("outputs/yolo_test")

print("\nAnnotated image saved in:")
print(output_path.resolve())
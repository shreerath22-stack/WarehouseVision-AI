from ultralytics import YOLO
from pathlib import Path

# Load our trained warehouse model
model = YOLO(r"models\warehouse_yolo11_final.pt")

# Validation image folder
image_folder = Path(r"data\processed\warehouse_yolo_clean\valid\images")

# Store total counts
total_counts = {}

print("\n📦 Warehouse Dataset Object Count")
print("=" * 40)

# Process every image
for image_path in image_folder.glob("*.jpg"):

    results = model.predict(
        source=str(image_path),
        conf=0.25,
        verbose=False
    )

    for result in results:
        for box in result.boxes:

            class_id = int(box.cls[0])
            object_name = model.names[class_id]

            total_counts[object_name] = (
                total_counts.get(object_name, 0) + 1
            )

print("\n📊 Total Detected Objects")
print("-" * 40)

for object_name, count in sorted(total_counts.items()):
    print(f"{object_name}: {count}")
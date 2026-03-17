from ultralytics import YOLO
import cv2

# Load pretrained vehicle detector
model = YOLO("yolov8n.pt")

def extract_traffic_state(image_path):
    img = cv2.imread(image_path)

    results = model(img)[0]

    north = south = east = west = 0

    h, w, _ = img.shape

    for box in results.boxes:
        cls = int(box.cls[0])

        # Vehicle class IDs in YOLO
        if cls in [2, 3, 5, 7]:  # car, bike, bus, truck
            x1, y1, x2, y2 = box.xyxy[0]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Determine which road region vehicle belongs to
            if cy < h / 2 and cx < w / 2:
                north += 1
            elif cy > h / 2 and cx < w / 2:
                south += 1
            elif cx > w / 2 and cy < h / 2:
                east += 1
            else:
                west += 1

    # Convert to DQN state format
    north_south = north + south
    east_west = east + west

    return [north_south, east_west], {
        "north": north,
        "south": south,
        "east": east,
        "west": west
    }
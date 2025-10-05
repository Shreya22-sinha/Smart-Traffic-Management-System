from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

def detect_image(image_path, socketio=None):
    image = cv2.imread(image_path)
    results = model(image)[0]
    vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    count = sum(1 for c in results.boxes.cls if int(c.item()) in vehicle_classes)
    light = "green" if count <= 5 else "yellow" if count <= 10 else "red"
    wait_time = max(5, min(60, 80 - count))

    data = {
        'vehicle_count': count,
        'light_status': light,
        'wait_time': wait_time
    }

    if socketio:
        socketio.emit('update', data)

    return data  # ✅ Return the result for the HTTP response

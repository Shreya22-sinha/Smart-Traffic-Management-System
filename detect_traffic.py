from ultralytics import YOLO
import cv2
from dqn_agent import DQNAgent

# Load YOLO
model = YOLO("yolov8n.pt")

# Initialize DQN Agent
agent = DQNAgent()

def detect_image(image_path, socketio=None):
    image = cv2.imread(image_path)
    results = model(image)[0]

    vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    count = sum(1 for c in results.boxes.cls if int(c.item()) in vehicle_classes)

    # -------------------------------
    # DQN STATE
    # -------------------------------
    state = [count, 0]

    # DQN chooses action
    action = agent.choose_action(state)

    # -------------------------------
    # ACTION → signal timing
    # -------------------------------
    wait_times = [10, 20, 30]
    wait_time = wait_times[action]

    light_map = ["green", "yellow", "red"]
    light = light_map[action]

    # -------------------------------
    # REWARD FUNCTION
    # -------------------------------
    reward = -count  # fewer vehicles = better

    next_state = [count, wait_time]

    # Train DQN
    agent.store(state, action, reward, next_state)
    agent.train()

    # -------------------------------
    # Output data
    # -------------------------------
    data = {
        'vehicle_count': count,
        'light_status': light,
        'wait_time': wait_time
    }

    # Send to frontend
    if socketio:
        socketio.emit('update', data)

    return data

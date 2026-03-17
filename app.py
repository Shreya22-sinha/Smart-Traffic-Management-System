import os
import subprocess
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# Import DQN Agent
from dqn_agent import DQNAgent

# ===============================
# 🚦 CROSSROAD TRAFFIC ANALYZER
# ===============================
def analyze_crossroad(boxes, image_width, image_height):
    directions = {
        "north": 0,
        "south": 0,
        "east": 0,
        "west": 0
    }

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # Determine direction based on position
        if cy < image_height / 2:
            directions["north"] += 1
        else:
            directions["south"] += 1

        if cx < image_width / 2:
            directions["west"] += 1
        else:
            directions["east"] += 1

    return directions

# -------- Free Port 5005 Automatically --------
try:
    result = subprocess.check_output('netstat -ano | findstr :5005', shell=True).decode()
    for line in result.splitlines():
        pid = line.strip().split()[-1]
        os.system(f'taskkill /PID {pid} /F')
except Exception:
    pass

# -------- YOLO Model Setup --------
model = YOLO("yolov8n.pt")
print("Device set to use", model.device)

# -------- Initialize DQN Agent --------
agent = DQNAgent()

# Store last decision (needed for feedback learning)
last_state = None
last_action = None

# -------- AI Learning Stats --------
total_feedback = 0
good_feedback = 0

# -------- Flask App Setup --------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# -------- Routes --------

@app.route('/')
def home():
    return render_template('index2.html')


# ===============================
# 🚦 DETECTION + DQN DECISION
# ===============================
@app.route('/detect', methods=['POST'])
def detect():
    global last_state, last_action

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # 🔹 Run YOLO detection
    results = model(filepath)
    boxes = results[0].boxes

    image = results[0].orig_img
    h, w, _ = image.shape

    vehicle_boxes = []

    for box in boxes:
       cls = int(box.cls)
       label = model.names[cls]
       if label.lower() in ['car', 'bus', 'truck', 'motorbike']:
           vehicle_boxes.append(box)

    # Count vehicles per direction
    directions = analyze_crossroad(vehicle_boxes, w, h)

    vehicle_count = sum(directions.values())

    # ----------------------------
    # 🤖 DQN TRAFFIC CONTROL
    # ----------------------------
    state = [vehicle_count, 0]

    action = agent.choose_action(state)

    # Map actions to signal times
    wait_times = [10, 20, 30]
    wait_time = wait_times[action]

    light_map = ["green", "yellow", "red"]
    light_status = light_map[action]

    # Store for RLHF feedback
    last_state = state
    last_action = action

    # Reward for automatic learning
    reward = -vehicle_count
    next_state = [vehicle_count, wait_time]

    agent.store(state, action, reward, next_state)
    agent.train()

    # ----------------------------
    # 🧠 Simple Advisory (LLM style)
    # ----------------------------
    if vehicle_count == 0:
       advisory = "Road is clear. No congestion."
    else:
       # Find most congested direction
       worst_road = max(directions, key=directions.get)

       advisory_map = {
           "north": "👉 Prioritize the north-south road",
           "south": "👉 Extend green signal for south road",
           "east": "👉 Allow right road to clear congestion",
           "west": "👉 Allow left road to clear congestion"
        }
       advisory = advisory_map[worst_road]

    # ----------------------------
    # Calculate AI confidence
    # ----------------------------
    confidence = 0
    if total_feedback > 0:
        confidence = int((good_feedback / total_feedback) * 100)

    # ----------------------------
    # Prepare response
    # ----------------------------
    result = {
        'vehicle_count': vehicle_count,
        'wait_time': wait_time,
        'light_status': light_status,
        'llm_output': advisory,
        'directions': directions,
        'confidence': confidence,
        'total_feedback': total_feedback
    }

    socketio.emit('update', result)

    return jsonify(result)


# ===============================
#  RLHF HUMAN FEEDBACK ROUTE
# ===============================
@app.route('/feedback', methods=['POST'])
def feedback():
    global last_state, last_action
    global total_feedback, good_feedback

    data = request.json
    feedback_type = data.get('feedback')

    # Update feedback counters
    total_feedback += 1
    if feedback_type == "good":
        good_feedback += 1

    reward = 10 if feedback_type == "good" else -10

    if last_state is not None:
        agent.store(last_state, last_action, reward, last_state)
        agent.train()

    # Calculate confidence %
    confidence = 0
    if total_feedback > 0:
        confidence = int((good_feedback / total_feedback) * 100)

    return jsonify({
        "status": "feedback received",
        "confidence": confidence,
        "total_feedback": total_feedback
    })


# -------- Run App --------
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5005, debug=True, use_reloader=False)


from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
import os

from detect_traffic import detect_image  # ✅ Import your real detection

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/detect', methods=['POST'])
def detect():
    image = request.files['image']
    filename = secure_filename(image.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath)

    # ✅ Use actual YOLO detection
    result = detect_image(filepath, socketio)

    return jsonify({
        'status': 'success',
        'image_url': f'/static/uploads/{filename}',
        **result  # This spreads vehicle_count, light_status, wait_time
    })

if __name__ == '__main__':
    socketio.run(app, debug=True,port=5001,use_reloader=False)

# server.py
from flask import Flask,render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import cv2
from functional.real_time_prediction import predict_frame, initialization

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

model = None
hands = None
mp_hands = None
mp_drawing = None

initialization('RandomForest')

@app.route('/translator')
def translator():
    return render_template('translator.html')

@socketio.on('frame')
def handle_frame(data):
    image_b64 = data.get('image')
    if not image_b64:
        return
    try:
        b64_data = image_b64.split(',')[1]
    except IndexError:
        print("Invalid image data")
        return
    img_bytes = base64.b64decode(b64_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    frame = cv2.flip(frame,1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    prediction = predict_frame(frame, image_rgb)
    cv2.putText(frame, str(prediction), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    emit('processed_frame', {
    'image': f"data:image/jpeg;base64,{jpg_as_text}"
        })

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port = 5000)

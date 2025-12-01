from flask import Flask, Response, jsonify, request, send_file
from functional.real_time_prediction import predict_frame, initialization, prediction_generator
import numpy as np
import mediapipe as mp
import cv2
import io
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
initialization('RandomForest')

@app.route("/video_prediction")
def video_prediction():
    return Response(
        prediction_generator(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route("/predict_frame", methods=["POST"])
def predict_frame_api():
    if 'image' not in request.files:
        return {"error": "No image uploaded"}, 400

    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    val = predict_frame(frame, image)

    cv2.putText(frame, str(val), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        return {"error": "Failed to encode frame"}, 500

    return send_file(
        io.BytesIO(buffer.tobytes()),
        mimetype='image/jpeg'
    )


@app.route("/status")
def status():
    return jsonify({"status": "API is running"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
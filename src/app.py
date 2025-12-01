from flask import Flask, Response, jsonify, request
from functional.real_time_prediction import predict_frame, initialization, prediction_generator
import numpy as np
import mediapipe as mp
import cv2

app = Flask(__name__)
initialization('RandomForest')

@app.route("/video_prediction")
def video_prediction():
    return Response(
        prediction_generator(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    result = predict_frame(frame, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return jsonify({'prediction': str(result)})


@app.route("/status")
def status():
    return jsonify({"status": "API is running"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
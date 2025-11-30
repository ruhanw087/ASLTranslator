from flask import Flask, Response, jsonify
from functional.real_time_prediction import run_classification, initialization, prediction_generator
import joblib
import mediapipe as mp
import cv2

app = Flask(__name__)
initialization('RandomForest')

def video_prediction():
    return Response(
        prediction_generator(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route("/status")
def status():
    return jsonify({"status": "API is running"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
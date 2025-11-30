from flask import Flask, Response, render_template
from functional.real_time_prediction import run_classification, initialization, prediction_generator
import joblib
import mediapipe as mp
import cv2

app = Flask(__name__)
initialization('RandomForest')

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/video_prediction")
def video_prediction():
    return Response(prediction_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=False)
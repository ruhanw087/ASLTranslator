import os
import cv2
import joblib
import mediapipe as mp
import sklearn

def run_classification(model_name):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    model = joblib.load(f"src/functional/{model_name}model.pkl")
    cap = cv2.VideoCapture()
    if not cap.isOpened():
        print("Camera failed to open")
        return
    while True:
        ret,frame = cap.read()
        if not ret:
            print("Failed to capture video")
            break

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key==ord('q'):
            print("Quitting Successfully")
        

        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_images()

import os
import cv2
import joblib
import mediapipe as mp
import sklearn
model = None
hands = None
mp_hands = None
mp_drawing = None

def initialization(model_name):
    global model
    model = joblib.load(f"src/functional/{model_name}.pkl")
    global hands
    global mp_hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.2
    )
    global mp_drawing
    mp_drawing = mp.solutions.drawing_utils

def predict_frame(frame,image):
    hand = []
    results = hands.process(image)
    if results.multi_hand_landmarks:
        for each_hand in results.multi_hand_landmarks:
                
            mp_drawing.draw_landmarks(
                frame,
                each_hand, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=5, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255,30, 30), thickness=5, circle_radius=8)
            )
            
            for landmark in each_hand.landmark:
                hand.append(landmark.x)
                hand.append(landmark.y)
                hand.append(landmark.z)
                    
        value = model.predict([hand[0:63]])
        return value[0]
    else:
        return "waiting for hands"


def run_classification(model_name):
    initialization(model_name)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera failed to open")
        return
    while True:
        ret,frame = cap.read()
        if not ret:
            print("Failed to capture video")
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        val = predict_frame(frame,image)
        cv2.putText(frame, str(val), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key==ord('q'):
            print("Quitting Successfully")
            break
    cap.release()
    cv2.destroyAllWindows()

def prediction_generator():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("cap not opened")
            return
        while True:
            ret,frame = cap.read()
            frame = cv2.flip(frame,1)
            if not ret:
                print("ret not opened")
                raise Exception("Failed to encode frame")
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            val = predict_frame(frame,image)
            cv2.putText(frame, str(val), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
            
            ret2, buffer = cv2.imencode('.jpg', frame)
            #frame2 = buffer.tobytes()
            frame2 = buffer.tobytes()
            yield (b'--frame2\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')
    except Exception as e:

        raise e

if __name__ == "__main__":
    run_classification("RandomForest")

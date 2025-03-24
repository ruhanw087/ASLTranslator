import os
import cv2
import joblib
import mediapipe as mp
import sklearn

def run_classification(model_name):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    model = joblib.load(f"src/functional/{model_name}model.pkl")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Camera failed to open")
        return
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.2
        )
    prev_value = ''
    while True:
        hand = []
        ret,frame = cap.read()
        if not ret:
            print("Failed to capture video")
            break

  


        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                    
            value = model.predict([hand[0:42]])

            print(value[0])
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF


        if key==ord('q'):
            print("Quitting Successfully")
            break

        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_classification("RandomForest")

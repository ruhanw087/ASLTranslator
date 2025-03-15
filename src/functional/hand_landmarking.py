import mediapipe as mp
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib

def create_landmarks():
    matplotlib.use("TkAgg")

    img_data_dir = "src/data/manual_images"

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.2
        )

    for sub_dir in os.listdir(img_data_dir):
        for path in os.listdir(os.path.join(img_data_dir,sub_dir))[:1]:
            src = cv2.imread(os.path.join(img_data_dir,sub_dir,path))
            if src is None:
                print("Failed to load images")
            image = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            if results.multi_hand_landmarks:
                for each_hand in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        each_hand, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

            plt.figure()
            plt.imshow(image)

    plt.show()

if __name__ == "__main__":
    create_landmarks()


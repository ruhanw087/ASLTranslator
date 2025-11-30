import mediapipe as mp
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

def create_landmarks(d_type):
    matplotlib.use("TkAgg")

    
    img_data_dir = "src/data/manual_images" if d_type == "manual" else "src/data/ASL_Alphabet_Dataset/asl_alphabet_train"


    mp_hands = mp.solutions.hands
    columns=[]
    list_data = []
    list_labels = []
    for i in range(1,22):
        columns.append(f"x{i}")
        columns.append(f"y{i}")
        columns.append(f"z{i}")

    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.2
        )

    for sub_dir in os.listdir(img_data_dir):
        print(sub_dir)
        for path in os.listdir(os.path.join(img_data_dir,sub_dir)):
            src = cv2.imread(os.path.join(img_data_dir,sub_dir,path))
            if src is None:
                print("Failed to load images")
            image = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            if results.multi_hand_landmarks:
                for each_hand in results.multi_hand_landmarks:
                    '''
                    mp_drawing.draw_landmarks(
                        image,
                        each_hand, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )
                    '''
                    list_hand = []
                    list_labels.append(sub_dir)
                    for landmark in each_hand.landmark:
                        list_hand.append(landmark.x)
                        list_hand.append(landmark.y)
                        list_hand.append(landmark.z)
                    
                    list_data.append(list_hand)

    data_df = pd.DataFrame(list_data, columns = columns)
    return data_df, list_labels


def keep_first_300_files(img_data_dir):
    # Get list of all files in the directory
    for sub_dir in os.listdir(img_data_dir):
        count = 0
        print(sub_dir)
        for path in os.listdir(os.path.join(img_data_dir,sub_dir)):
            count+=1
            if count > 300:
                os.remove(os.path.join(img_data_dir,sub_dir,path))


    # Delete files beyond the first 300


def write_to_csv(values):
    if isinstance(values,list):
        pd.DataFrame(values,columns=['labels']).to_csv("src/data/landmark_labels.csv", index=False)
    else:
        values.to_csv("src/data/landmarks.csv", index=False)

if __name__ == "__main__":
    #keep_first_300_files("src/data/ASL_Alphabet_Dataset/asl_alphabet_train")
    data, labels = create_landmarks("automatic")
    write_to_csv(data)
    write_to_csv(labels)


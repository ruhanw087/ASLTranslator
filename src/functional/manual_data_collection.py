import os
import cv2

print(cv2.__version__)

def collect_images():
    base_dir= "src/data/manual_images"
    sub_dir = ['1','2','3']
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    for sub in sub_dir:
        path = os.path.join(base_dir, sub)
        if not os.path.exists(path):
            os.makedirs(path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera failed to open")
        return
    
    print("press r to start collecting images")
        
    count = {sub: 0 for sub in sub_dir}
    key_list = list(count.keys())
    i = 0
    while True:
        ret,frame = cap.read()
        if not ret:
            print("Failed to capture video")
            break

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key==ord('r'):
            print(f"Capturing next 100 frames for {key_list[i]}")
            while count[key_list[i]] <100:
                ret1,frame1 = cap.read()
                cv2.waitKey(1)
                cv2.imshow("Frame", frame1)
                img_path = os.path.join(base_dir, key_list[i], f"image_{count[key_list[i]]:03d}.jpg")
                cv2.imwrite(img_path, frame1)
                count[key_list[i]] += 1
            print("100 Images Captured")
            i+=1
        if all(c>=100 for c in count.values()):
            print("All sub directories full")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Video stopped")
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_images()


    
        


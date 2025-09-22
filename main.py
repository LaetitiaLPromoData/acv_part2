import cv2
import mediapipe as mp
from pathlib import Path
import os
import pandas as pd

drawing_utils = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles
Holistic = mp.solutions.holistic.Holistic

def worflow_mediapipe(mediapipe_model,mediapipe_based_filter):
    """Run a media pipe model on each video frame grabbed by the webcam and draw results on it

    Args:
        mediapipe_model (): A mediapipe model
        mediapipe_based_filter (): a function to draw model results on frame

    Returns:
        np.ndarray, mediapipe model result
    """
    cap = cv2.VideoCapture(0)

    try:
        with mediapipe_model as model:
            while cap.isOpened():
                success, image = cap.read()
            
                if not success:
                    print("Ignoring empty camera frame.")
                    continue
        
                try:
                    results = model.process(image)
                except Exception:
                    results = None
                
                if results and results.pose_landmarks:
                    result_image = mediapipe_based_filter(image, results)
                else:
                    result_image = image

                cv2.imshow('MediaPipe', result_image)

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()           

    return image, results


def draw_holistic_results(image, results):
    
    # drawing_utils.draw_landmarks(
    #     image,
    #     results.left_hand_landmarks,
    #     mp.solutions.holistic.HAND_CONNECTIONS,
    #     connection_drawing_spec=drawing_styles.get_default_hand_connections_style()
    # )

    drawing_utils.draw_landmarks(
        image,
        results.pose_landmarks,
        mp.solutions.holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style()
    )
    
    return image


def draw_selected_points(image, results, indices):
    if results.pose_landmarks:
        h,w, _ = image.shape
        landmarks = results.pose_landmarks.landmark

        for idx in indices:
            lm = landmarks[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 10, (255, 0, 0), -1)  # bleu

    return image


def extract_pose(image, model, indices):
    """Run a media pipe model on one image and return coordinated of specific point

    Args:
        image : the image to process
        mediapipe_model (): A mediapipe model
        indices : a list of point to find in the image

    Returns:
        np.ndarray, 0
    """
    
    try:
        results = model.process(image)
        list_pos = []
        for idx in indices:
            landmarks = results.pose_landmarks.landmark
            lm = landmarks[idx]
            list_pos.append((lm.x, lm.y,lm.z))
    except Exception:
        list_pos = 0
        print("Error")
    if (list_pos != 0):
        if len(list_pos) == len(indices):
            return list_pos
        else: return 0
    else: return 0

def process_image(folder_path, mediapipe_model, indices):
    #creation du dataframe
    columns = [f"{axis}_{i}" for i in range(len(indices)) for axis in ['x','y','z']] + ["target"]
    database = pd.DataFrame(columns=columns)

    for i, name_folder in enumerate(["Bas","Haut","Autres"]):
        sub_folder_path = os.path.join(folder_path, name_folder)

        for image_name in Path(sub_folder_path).glob("*.jpg"):
            # image_path = os.path.join(sub_folder_path, image_name)
            image = cv2.imread(image_name)
            list_coord = extract_pose(image, mediapipe_model, indices)
            if list_coord != 0: # a détecter ts les points dans l'image
                flat_list = [coord for landmark in list_coord for coord in landmark]
                flat_list.append(i)
                database.loc[len(database)] = flat_list  
    
    database.to_csv("database.csv")
    return database

        
def main():
    mediapipe_model=Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    indices = [12,14,16,11,13,15,23,24]
    image_folder = Path("Images-pompes")
    process_image(image_folder,mediapipe_model,indices)



    # list = extract_pose(image,mediapipe_model,indices)
    print(list)
#     last_image, last_results = worflow_mediapipe(
#     mediapipe_model=mediapipe_model,
#     mediapipe_based_filter = lambda img, res: draw_selected_points(img, res, indices)
# )

if __name__ == "__main__":
    main()

import os
import sys
import pandas as pd
from datetime import time
from tensorflow.python.keras.callbacks_v1 import TensorBoard
import yolo as y


# To change the initial path here before executing
def get_parent_dir(n=1):
    """ returns the nth parent of the current directory """

    cpath = os.path.dirname(os.path.join(os.environ['PROJECT_PATH'], "Badminton_Analysis.ipynb"))
    for k in range(n):
        cpath = os.path.dirname(cpath)
    return cpath

def create_model(train_weight_final,anchors_path,yolo_classname, vpath):
    print("A")
    score = 0.25
    num_gpu = 1
    yolo = y.YOLO(
        **{
            "model_path": train_weight_final,
            "anchors_path": anchors_path,
            "classes_path": yolo_classname,
            "score": score,
            "gpu_num": num_gpu,
            "model_image_size": (416, 416),
        }
    )
    print("B")
    out_df = pd.DataFrame(
        columns=[
            "image",
            "image_path",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "label",
            "confidence",
            "x_size",
            "y_size",
        ]
    )
    print("C")
    # labels to draw on images
    class_file = open(yolo_classname, "r")
    print("D")
    input_labels = [line.rstrip("\n") for line in class_file.readlines()]
    print("Found {} input labels: {} ...".format(len(input_labels), input_labels))

    y.detect_video(yolo, vpath, output_path="testPA8")
    print("D")
    yolo.close_session()
    print("E")

def videotoclips():
    clips = []

    return clips

def main():
    os.environ['PROJECT_PATH'] = "."

    src_path = os.path.join(get_parent_dir(0))
    sys.path.append(src_path)

    Yolo_Model_Folder = os.path.join(get_parent_dir(0), "YoloV3")
    YOLO_classname = os.path.join(Yolo_Model_Folder, "data_classes.txt")

    YOLO_Folder = os.path.join(get_parent_dir(0), "YoloV3")
    keras_path = os.path.join(YOLO_Folder, "keras_yolo3")
    anchors_path = os.path.join(keras_path, "yolo_anchors.txt")

    yolo_log_dir = os.path.join(Yolo_Model_Folder, "logs")
    train_weight_final = os.path.join(yolo_log_dir, "trained_weights_final.h5")


    ressources = os.path.join(get_parent_dir(0), "res")
    video = os.path.join(ressources, "video")
    videotest = os.path.join(video, "video.mp4")

    create_model(train_weight_final, anchors_path, YOLO_classname, videotest)


""" Data_Folder = os.path.join(get_parent_dir(0), "data")
    MXNet_Train_Folder = os.path.join(Data_Folder, "MXNetframes")
    Yolo_Train_Folder = os.path.join(Data_Folder, "YoloTrainingImages")
    YOLO_filename = os.path.join(Yolo_Train_Folder, "data_train.txt")
    train_weight_stage = os.path.join(yolo_log_dir, "trained_weights_stage_1.h5")
    yolo_train_weights = os.path.join(keras_path, "yolo.h5")
    image_test_folder = os.path.join(Data_Folder, "Test")
    detection_results_folder = os.path.join(Data_Folder, "Test_Results")
    detection_results_file = os.path.join(detection_results_folder, "Yolo_Detection_Results.csv")
    yolo_test_weights = os.path.join(Yolo_Model_Folder, "logs", "trained_weights_final.h5")"""

if __name__ == '__main__':
    main()
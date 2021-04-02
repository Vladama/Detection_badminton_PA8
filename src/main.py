import os
import sys
import pandas as pd
import json

from timeit import default_timer as timer

import yolo as y
import glob

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from datetime import datetime

import pandas as pd


# To change the initial path here before executing
def get_parent_dir(n=1):
    """ returns the nth parent of the current directory """

    cpath = os.path.dirname(os.path.join(os.environ['PROJECT_PATH'], "main.py"))
    for k in range(n):
        cpath = os.path.dirname(cpath)
    return cpath

def create_model(train_weight_final,anchors_path,yolo_classname, vpath, timecode):

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

    out_df = pd.DataFrame(
        columns=[
            "cut_nb",
            "frame_ID",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "label",
            "confidence",
        ]
    )
    df = out_df
    onoff = False
    cut_nb = 0
    index = 0
    for i in timecode:
        if onoff:
            end = i['time']
            end = end['secondes']
            onoff = False
            cut_nb += 1
            videotoclips(vpath, start, end, cut_nb)
            # labels to draw on images
            class_file = open(yolo_classname, "r")

            input_labels = [line.rstrip("\n") for line in class_file.readlines()]
            print("Found {} input labels: {} ...".format(len(input_labels), input_labels))

            df, index = y.detect_video(yolo, "res/img/video_" + str(cut_nb) + ".mp4", df, cut_nb, index,
                                      #output_path="res/img/video_detect" + str(cut_nb) + ".mp4",
                                        Video_on=False)

            os.remove("res/img/video_" + str(cut_nb) + ".mp4")

        if i['category'] == 'CAMERA':
            if i['description'] == 'BAD_camera vert':
                start = i['time']
                start = start['secondes']
                onoff = True

    yolo.close_session()
    return df

def videotoclips(vpath, start, end, cut_nb):

    start = float(start)
    end = float(end)
    cut = ffmpeg_extract_subclip(vpath, start, end, targetname="res/img/video_" + str(cut_nb) + ".mp4")
    return cut

def transform_df():
    pass

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
    text = os.path.join(ressources, "txt")
    jsoninfo = os.path.join(text, "games1.json")
    video = os.path.join(ressources, "video")
    videotest = os.path.join(video, "Video1.mp4")

    with open(jsoninfo) as json_data:
        data_dict = json.load(json_data)
        match = data_dict['match']
        timecode = match['actions']

    df = create_model(train_weight_final, anchors_path, YOLO_classname, videotest, timecode)
    df.to_csv("data_cut.csv", index=False)


if __name__ == '__main__':
    main()

import os
import sys
import json
import yolo as y
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import pandas as pd


# To change the initial path here before executing
def get_parent_dir(n=1):
    """ returns the nth parent of the current directory """

    cpath = os.path.dirname(os.path.join(os.environ['PROJECT_PATH'], "main.py"))
    for k in range(n):
        cpath = os.path.dirname(cpath)
    return cpath

def create_model(train_weight_final,anchors_path,yolo_classname, vpath, timecode):

    #create the model
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

    # create the dataframe
    df = pd.DataFrame(
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

    onoff = False
    cut_nb = 0
    index = 0

    # brut force en fonction du json des times codes

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

            df, index, width, height = y.detect_video(yolo, "res/img/video_" + str(cut_nb) + ".mp4", df, cut_nb, index,
                                      #output_path="res/img/video_detect" + str(cut_nb) + ".mp4",
                                        Video_on=False)

            os.remove("res/img/video_" + str(cut_nb) + ".mp4")

        if i['category'] == 'CAMERA':
            if i['description'] == 'BAD_camera vert':
                start = i['time']
                start = start['secondes']
                onoff = True

    yolo.close_session()

    df = transform_df(df, width, height)

    return df

# Function that create clips of the initial video based on timecode given in back

def videotoclips(vpath, start, end, cut_nb):

    start = float(start)
    end = float(end)
    cut = ffmpeg_extract_subclip(vpath, start, end, targetname="res/img/video_" + str(cut_nb) + ".mp4")
    return cut


#Function that process the dataset correctly and call over function below

def transform_df(df, width, height):

    df = df[df.label != 2]

    df = out_duplicate(df)

    df = out_proportion(df, width, height)

    df['Xposition'] = (df['xmin'] + df['xmax']) // 2
    df['Yposition'] = (df['ymin'] + (df['ymax'] - df['ymin']) * (90/100))

    my_columns = ['cut_nb', 'frame_ID', 'xmin', 'xmax', 'Xposition', 'ymin', 'ymax', 'Yposition', 'label', 'confidence']
    df = df[my_columns]

    return df


# Function that delete the duplicate

def out_duplicate(df):
    
    for i in range(len(df)):

        if i == (df.shape[0] - 1):
            break

        if df['frame_ID'][i] == df['frame_ID'][i + 1]:

            if (df['label'][i] == df['label'][i + 1]):

                if (df['label'][i] == 0):

                    if (df['ymax'][i] > df['ymax'][i + 1]):

                        df.drop(labels=i + 1, axis=0, inplace=True)
                        df = df.reset_index(drop=True)

                    else:
                        df.drop(labels=i, axis=0, inplace=True)
                        df = df.reset_index(drop=True)

                else:

                    if (df['ymax'][i] < df['ymax'][i + 1]):

                        df.drop(labels=i + 1, axis=0, inplace=True)
                        df = df.reset_index(drop=True)

                    else:

                        df.drop(labels=i, axis=0, inplace=True)
                        df = df.reset_index(drop=True)

    return df


# Function that clear the box that are out of proportion

def out_proportion(df, width, height):

    width = width * (25/100)
    height = height * (60/100)

    for i in range(len(df)):

        if i == (df.shape[0] - 1):
            break

        x = df['xmax'][i] - df['xmin'][i]
        y = df['ymax'][i] - df['ymin'][i]

        if (x > width) or (y > height):
            df.drop(labels=i, axis=0, inplace=True)

    df = df.reset_index(drop=True)
    
    return df


def main():
    os.environ['PROJECT_PATH']='.'

    src_path = os.path.join(get_parent_dir(0))# path src
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
    jsoninfo = os.path.join(text, "games1.json") # json de cut (camera on/off)
    video = os.path.join(ressources, "video")
    videotest = os.path.join(video, "Video1.mp4") # vidéo de référence

    with open(jsoninfo) as json_data: ### Utile que si tu me renvoie le json complet et pas directement la partie action
        data_dict = json.load(json_data)
        match = data_dict['match']
        timecode = match['actions'] ### jusqu'à la

    df = create_model(train_weight_final, anchors_path, YOLO_classname, videotest, timecode) # function that run all the process
    df.to_csv("data_final.csv", index=False) # return csv


if __name__ == '__main__':
    main()

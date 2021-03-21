import os
import numpy as np

from keras.layers import Input
from timeit import default_timer as timer
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# """YOLO_v3 Model Defined in Keras."""

from functools import wraps

import tensorflow as tf
from keras import backend as K
from keras.layers import (
    Conv2D,
    Add,
    ZeroPadding2D,
    UpSampling2D,
    Concatenate,
    MaxPooling2D,
)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

import colorsys
from PIL import Image, ImageFont, ImageDraw
import os
import cv2
from functools import reduce


# To change the initial path here before executing
def get_parent_dir(n=1):
    """ returns the nth parent of the current directory """

    cpath = os.path.dirname(os.path.join(os.environ['PROJECT_PATH'], "main.py"))
    for k in range(n):
        cpath = os.path.dirname(cpath)
    return cpath


# CLass Yolo - base class for Yolo V3

class YOLO(object):
    _defaults = {
        "model_path": "models/YoloV3/keras_yolo3/yolo.h5",
        "anchors_path": "models/YoloV3/keras_yolo3/yolo_anchors.txt",
        "classes_path": "models/YoloV3/data_classes.txt",
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(clls, n):
        if n in clls._defaults:
            return clls._defaults[n]
        else:
            return "Unrecognized attribute '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        #         self.sess = K.get_session()

        self.config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
                                               # device_count = {'GPU': 1}
                                               )
        self.config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=self.config)
        tf.compat.v1.keras.backend.set_session(self.sess)

        # self.sess = tf.compat.v1.keras.backend.get_session()
        tf.compat.v1.disable_eager_execution()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_name = f.readlines()
        class_name = [c.strip() for c in class_name]
        return class_name

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(",")]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith(".h5"), "Keras model or weights must be a .h5 file."

        # Load model, or construct model and load weights.
        start = timer()
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = tf.keras.models.load_model(model_path, compile=False)
        except:
            self.yolo_model = (
                tiny_yolo_body(
                    Input(shape=(None, None, 3)), num_anchors // 2, num_classes
                )
                if is_tiny_version
                else yolo_body(
                    Input(shape=(None, None, 3)), num_anchors // 3, num_classes
                )
            )
            self.yolo_model.load_weights(
                self.model_path
            )  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == num_anchors / len(
                self.yolo_model.output
            ) * (
                           num_classes + 5
                   ), "Mismatch between model and given anchor and class sizes"

        end = timer()
        print(
            "{} model, anchors, and classes loaded in {:.2f}sec.".format(
                model_path, end - start
            )
        )

        # Generate colors for drawing bounding boxes.
        if len(self.class_names) == 1:
            self.colors = ["GreenYellow"]
        else:
            hsv_tuples = [
                (x / len(self.class_names), 1.0, 1.0)
                for x in range(len(self.class_names))
            ]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(
                map(
                    lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                    self.colors,
                )
            )
            np.random.seed(10101)  # Fixed seed for consistent colors across runs.
            np.random.shuffle(
                self.colors
            )  # Shuffle colors to decorrelate adjacent classes.
            np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(
            self.yolo_model.output,
            self.anchors,
            len(self.class_names),
            self.input_image_shape,
            score_threshold=self.score,
            iou_threshold=self.iou,
        )
        return boxes, scores, classes

    def detect_image(self, image, frame_nb, show_stats=True, Video_on=False):
        start = timer()
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, "Multiples of 32 required"
            assert self.model_image_size[1] % 32 == 0, "Multiples of 32 required"
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (
                image.width - (image.width % 32),
                image.height - (image.height % 32),
            )
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype="float32")
        if show_stats:
            print(image_data.shape)
        image_data /= 255.0
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                tf.compat.v1.keras.backend.learning_phase(): 0,
            },
        )
        #         out_boxes=out_boxes.ref().deref()
        #         out_scores=out_scores.ref().deref()
        #         out_classes=out_classes.ref().deref()
        if show_stats:
            print("Found {} boxes for {}".format(len(out_boxes), "img"))
        out_prediction = []
        if Video_on:
            keras_path = os.path.join(get_parent_dir(0), "YoloV3", "keras_yolo3")
            font_path = os.path.join(keras_path, "font/FiraMono-Medium.otf")
            # print("font",font_path)
            font = ImageFont.truetype(
                font=font_path, size=np.floor(3e-2 * image.size[1] + 0.5).astype("int32")
            )
            thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            if Video_on:

                label = "{} {:.2f}".format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype("int32"))
            left = max(0, np.floor(left + 0.5).astype("int32"))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype("int32"))
            right = min(image.size[0], np.floor(right + 0.5).astype("int32"))

            # image was expanded to model_image_size: make sure it did not pick
            # up any box outside of original image (run into this bug when
            # lowering confidence threshold to 0.01)
            if top > image.size[1] or right > image.size[0]:
                continue
            if show_stats:
                print(label, (left, top), (right, bottom))

            # output as xmin, ymin, xmax, ymax, class_index, confidence
            out_prediction.append([frame_nb, left, top, right, bottom, c, score])

            if Video_on:
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, bottom])

                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i], outline=self.colors[c]
                    )
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[c],
                )

                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                del draw

        end = timer()
        if show_stats:
            print("Time spent: {:.3f}sec".format(end - start))
        return out_prediction, image

    def close_session(self):
        self.sess.close()


def detect_video(yolo, video_path, out_df, output_path="", Video_on=False):
    vid = cv2.VideoCapture(video_path)

    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    video_FourCC = cv2.VideoWriter_fourcc(*"mp4v")  # int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (
        int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    isOutput = True if output_path != "" else False
    if isOutput:
        print(
            "Processing {} with frame size {} at {:.1f} FPS".format(
                os.path.basename(video_path), video_size, video_fps
            )
        )
        # print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        if Video_on:
            out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    frame_nb = 1
    index = 0
    while vid.isOpened():
        return_value, frame = vid.read()
        if not return_value:
            break
        # opencv images are BGR, translate to RGB
        frame = frame[:, :, ::-1]
        image = Image.fromarray(frame)
        out_pred, image = yolo.detect_image(image, frame_nb, show_stats=False, Video_on=True)
        frame_nb += 1
        for i in out_pred:
            out_df.loc[index] = i
            index += 1
        if Video_on:
            result = np.asarray(image)
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(
                result,
                text=fps,
                org=(3, 15),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.50,
                color=(255, 0, 0),
                thickness=2,
            )

            if isOutput:
                out.write(result[:, :, ::-1])

    if Video_on:
        vid.release()
        out.release()
        return out_df

    else:
        return out_df


##Yolo class ends


# """YOLO_v3 Model Defined in Keras."""

@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_c_kwargs = {"kernel_regularizer": l2(5e-4)}
    darknet_c_kwargs["padding"] = (
        "valid" if kwargs.get("strides") == (2, 2) else "same"
    )
    darknet_c_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_c_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_biases_kwargs = {"use_bias": False}
    no_biases_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_biases_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
    )


def resblock_body(x1, num_filters, num_blocks):
    """A series of resblocks starting with a downsampling Convolution2D"""
    # Darknet uses left and top padding instead of 'same' mode
    x1 = ZeroPadding2D(((1, 0), (1, 0)))(x1)
    x1 = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x1)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)),
        )(x1)
        x1 = Add()([x1, y])
    return x1


def darknet_body(x1):
    """Darknent body having 52 Convolution2D layers"""
    x1 = DarknetConv2D_BN_Leaky(32, (3, 3))(x1)
    x1 = resblock_body(x1, 64, 1)
    x1 = resblock_body(x1, 128, 2)
    x1 = resblock_body(x1, 256, 8)
    x1 = resblock_body(x1, 512, 8)
    x1 = resblock_body(x1, 1024, 4)
    return x1


def make_last_layers(x1, num_filter, out_filters):
    """6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer"""
    x1 = compose(
        DarknetConv2D_BN_Leaky(num_filter, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filter * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filter, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filter * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filter, (1, 1)),
    )(x1)
    y = compose(
        DarknetConv2D_BN_Leaky(num_filter * 2, (3, 3)),
        DarknetConv2D(out_filters, (1, 1)),
    )(x1)
    return x1, y


def yolo_body(input, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(input, darknet_body(input))
    x1, y1 = make_last_layers(darknet.output, 512, num_anchors * (num_classes + 5))

    x1 = compose(DarknetConv2D_BN_Leaky(256, (1, 1)), UpSampling2D(2))(x1)
    x1 = Concatenate()([x1, darknet.layers[152].output])
    x1, y2 = make_last_layers(x1, 256, num_anchors * (num_classes + 5))

    x1 = compose(DarknetConv2D_BN_Leaky(128, (1, 1)), UpSampling2D(2))(x1)
    x1 = Concatenate()([x1, darknet.layers[92].output])
    x1, y3 = make_last_layers(x1, 128, num_anchors * (num_classes + 5))

    return Model(input, [y1, y2, y3])


def tiny_yolo_body(input, num_anchors, num_classes):
    """Create Tiny YOLO_v3 model CNN body in keras."""
    x1 = compose(
        DarknetConv2D_BN_Leaky(16, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        DarknetConv2D_BN_Leaky(32, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        DarknetConv2D_BN_Leaky(64, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        DarknetConv2D_BN_Leaky(128, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        DarknetConv2D_BN_Leaky(256, (3, 3)),
    )(input)
    x2 = compose(
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        DarknetConv2D_BN_Leaky(512, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
        DarknetConv2D_BN_Leaky(1024, (3, 3)),
        DarknetConv2D_BN_Leaky(256, (1, 1)),
    )(x1)
    y1 = compose(
        DarknetConv2D_BN_Leaky(512, (3, 3)),
        DarknetConv2D(num_anchors * (num_classes + 5), (1, 1)),
    )(x2)

    x2 = compose(DarknetConv2D_BN_Leaky(128, (1, 1)), UpSampling2D(2))(x2)
    y2 = compose(
        Concatenate(),
        DarknetConv2D_BN_Leaky(256, (3, 3)),
        DarknetConv2D(num_anchors * (num_classes + 5), (1, 1)),
    )([x2, x1])

    return Model(input, [y1, y2])


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convertion of final layer features to bounding boxes."""
    num_anch = len(anchors)
    # Reshape to batch, height, width, num_anch, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anch, 2])

    grid_shape = K.shape(feats)[1:3]  # height, width
    y_grid = K.tile(
        K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1],
    )
    x_grid = K.tile(
        K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1],
    )
    grid = K.concatenate([x_grid, y_grid])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anch, num_classes + 5]
    )

    # Adjust preditions to each spatial grid point and anchor size.
    xy_box = (K.sigmoid(feats[..., :2]) + grid) / K.cast(
        grid_shape[::-1], K.dtype(feats)
    )
    wh_box = (
            K.exp(feats[..., 2:4])
            * anchors_tensor
            / K.cast(input_shape[::-1], K.dtype(feats))
    )
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, xy_box, wh_box
    return xy_box, wh_box, box_confidence, box_class_probs


def yolo_correct_boxes(xy_box, wh_box, input_shape, image_shape):
    yx_box = xy_box[..., ::-1]
    hw_box = wh_box[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(yx_box))
    image_shape = K.cast(image_shape, K.dtype(yx_box))
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2.0 / input_shape
    scale = input_shape / new_shape
    yx_box = (yx_box - offset) * scale
    hw_box *= scale

    box_mins = yx_box - (hw_box / 2.0)
    box_maxes = yx_box + (hw_box / 2.0)
    boxes = K.concatenate(
        [
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2],  # x_max
        ]
    )

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


# computing boxes and scores
def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    xy_box, wh_box, box_confidence, box_class_probs = yolo_head(
        feats, anchors, num_classes, input_shape
    )
    boxes = yolo_correct_boxes(xy_box, wh_box, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(
        yolo_outputs,
        anchors,
        num_classes,
        image_shape,
        max_boxes=20,
        score_threshold=0.6,
        iou_threshold=0.5,
):
    """Evaluate YOLO model on input and produce filtered boxes."""
    num_layers = len(yolo_outputs)
    anchor_mask = (
        [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    )  # default setting
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(
            yolo_outputs[l],
            anchors[anchor_mask[l]],
            num_classes,
            input_shape,
            image_shape,
        )
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype="int32")
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(tensor=boxes, mask=mask[:, c])
        class_box_scores = tf.boolean_mask(tensor=box_scores[:, c], mask=mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold
        )
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, "int32") * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


# Utility functions
def compose(*funcs):
    # Compose arbitrarily many functions, evaluated left to right.

    # Reference: https://mathieularose.com/function-composition-in-python/

    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError("Composition of empty sequence not supported.")


def letterbox_image(image, size):
    # resize image with unchanged aspect ratio using padding
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new("RGB", size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image

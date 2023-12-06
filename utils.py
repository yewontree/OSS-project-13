import cv2
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult, HandLandmarker
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import numpy as np
import logging, coloredlogs
import argparse
import pandas as pd
from datetime import datetime
from typing import TypeVar

from settings import *
import algorithm

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO", logger=logger)  # logger 설정, logger.debug() 함수로 로그메시지 표시


class DetectInfo:
    def __init__(self):
        self.detection_result = None  # 콜백(callback)함수 리턴값 저장용 변수 (callback 함수내에서 nonlocal로 access한다)
        self.detection_image = None
        self.detection_time = 0
        self.detection_time_pre = 0
        self.count = 0
        self.time_diff = self.time_total = self.time_diff_mean = 0

    def update(self, result):
        self.detection_result = result
        self.detection_time_pre, self.detection_time = self.detection_time, ms_timestamp_now()
        self.time_diff = self.detection_time - self.detection_time_pre
        if self.time_diff and self.detection_time_pre:
            self.count += 1
            self.time_total += self.time_diff
            self.time_diff_mean = self.time_total / self.count

    def update_live(self, result, output_image, timestamp_ms):
        self.detection_result = result
        self.detection_image = output_image.numpy_view()
        self.detection_time_pre, self.detection_time = self.detection_time, timestamp_ms
        self.time_diff = self.detection_time - self.detection_time_pre
        if self.time_diff and self.detection_time_pre:
            self.count += 1
            self.time_total += self.time_diff
            self.time_diff_mean = self.time_total / self.count

    def new_detection(self):
        return self.time_diff

    def reset_detection(self):
        self.time_diff = 0


# https://github.com/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb
def draw_fingercount_on_image(rgb_image, detection_result, algo=None, timestamp=None):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend(
            [landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style(),
        )

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # 핑커 카운터 알고리즘 수행
        if algo:  # algo(알고리즘 객체)가 None이 아니라면
            wrist, tip_fingers = algo.detect_tip_finger(np.array(hand_landmarks_proto.landmark))
        else:
            wrist, tip_fingers = None, [0, 0, 0, 0, 0]
            logger.warning(f"draw_fingercount_on_image() function called without algorithm !!!")
        finger_count = sum(tip_fingers)

        # Draw handedness (left or right hand) on the image.
        if finger_count:
            cv2.putText(
                annotated_image,
                text=f"{tip_fingers}",  # 검출 데이터 e.g. [1, 0, 1, 1, 0]
                org=(text_x, text_y),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=FONT_SIZE,
                color=HANDEDNESS_TEXT_COLOR,
                thickness=FONT_THICKNESS,
                lineType=cv2.LINE_AA,
            )

        if timestamp:
            caption = f"Algorithm: {algo.__class__.__name__}   {timestamp}"
        else:
            caption = f"Algorithm: {algo.__class__.__name__}"
        cv2.putText(
            annotated_image,
            text=caption,  # 알고리즘 명칭 표시
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=FONT_SIZE,
            color=(198, 222, 40),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    return annotated_image


# https://stackoverflow.com/questions/13520421/recursive-dotdict
class attrdict(dict):
    """
    a dictionary that supports dot(.) operation
    as well as dictionary access operation
    usage: d = attrdict() or d = attrdict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def extract_subclass(module, BaseClass) -> dict[str, TypeVar]:
    """extract module's derivative class from base-class

    Args:
        module (namespace): package module or file
        BaseClass (TypeVar): interface or base-class

    Returns:
        dict[str, TypeVar]: dictionary of str key & derived class
    """
    subclass = {}
    for attr in dir(module):
        try:
            v = getattr(module, attr)
            if issubclass(v, BaseClass) and v is not BaseClass:
                subclass[attr.lower()] = v  # str.lower() return a copy of the string converted to lowercase.
        except TypeError:
            continue
    return attrdict(subclass)


def get_args():
    algos = extract_subclass(algorithm, algorithm.Algorithm)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        help="select finger counter algorithm",
        default="basic",
        type=str,
        required=False,
        choices=list(algos.keys()),
    )
    parser.add_argument(
        "--log",
        help="select log level",
        default="info",
        type=str,
        required=False,
        choices=["debug", "info", "warning", "error", "critical"],
    )
    parser.add_argument(
        "--log_a",
        help="select algorithm log level",
        default="info",
        type=str,
        required=False,
        choices=["debug", "info", "warning", "error", "critical"],
    )
    parser.add_argument(
        "--save",
        help="save video data as .avi file",
        default=False,
        action="store_true",
        required=False,
    )
    args = parser.parse_args()  # get the command line arguments

    return args


def select_algorithm(algos, args):
    """returns the algorithm selected from the command line argument"""
    logger.info(f"'{algos[args.algo].__name__}' algorithm selected\n")
    return algos[args.algo]()


def ms_timestamp_now():
    now = datetime.now()
    timestamp_seconds = datetime.timestamp(now)  # Convert it to a timestamp (in seconds)
    return int(timestamp_seconds * 1000)


def to_datetime_str(time):
    # Convert seconds to a datetime object
    datetime_object = datetime.fromtimestamp(time)
    return datetime_object.strftime("%H:%M:%S")


def check_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"folder '{path}' created")


def get_video_writer(algo):
    check_folder(avi_folder)  # if not exist, make avi_folder
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # for saving video stream
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = avi_folder / f"{algo.__class__.__name__}({datetime_str}).avi"
    out = cv2.VideoWriter(str(filename), fourcc, 20.0, (640, 480))
    return out, filename


def save_data(results, image):
    if results:
        check_folder(data_folder)  # if not exist, make data_folder
        filename = data_folder / datetime.now().strftime("%Y-%m-%d_%H%M%S")
        cv2.imwrite(f"{filename}.jpg", image)
        logger.info(f"image saved: {filename}.jpg")

        data = []
        for i, result in enumerate(results):
            for mark in result:
                data.append([mark.x, mark.y, mark.z])
            pd.DataFrame(data, columns=["x", "y", "z"]).to_csv(f"{filename}_{i}.csv", index=False)
            logger.info(f"data saved: {filename}_{i}.csv")

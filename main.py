import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult, HandLandmarker
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import logging, coloredlogs

from settings import *  # loading settings.py data to current 'main' namespace

logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG", logger=logger)  # logger 설정, logger.debug() 함수로 로그메시지 표시


# https://github.com/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb
def draw_fingercount_on_image(rgb_image, detection_result):
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

        # Draw handedness (left or right hand) on the image.
        cv2.putText(
            annotated_image,
            f"{handedness[0].category_name}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated_image


def main():
    base_options = python.BaseOptions(model_asset_path="weights/hand_landmarker.task")
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector: HandLandmarker = vision.HandLandmarker.create_from_options(options)  # hand detector 객체 생성

    cap = cv2.VideoCapture(0)

    while True:
        success, image_origin = cap.read()
        if success:
            image = cv2.cvtColor(image_origin, cv2.COLOR_BGR2RGB)
            rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            detection_result: HandLandmarkerResult = detector.detect(rgb_frame)  # hand landmarker result (num_hands=2, 양손 가능)

            if detection_result:  # detection_result 로그데이터 생성
                logger.debug(f"detection_result:\n {detection_result}")

            annotated_image = draw_fingercount_on_image(  # 원본 카메라 영상에 detection 결과를 합성하는 듯~
                image_origin,
                detection_result,
            )

            cv2.imshow(f"Finger Counter", annotated_image)  # 동영상 재생
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            print(f"cap.read() error")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


# TODO 검출기가 출력하는 'HandLandmarkerResult' 데이터를 분석해야 함. detection_result: HandLandmarkerResult = detector.detect(rgb_frame)
# 아래 샘플 데이터는 로그 메시지로부터 추출함.
# notebook/hand_landmarker.ipynb 파일에서 분석할 예정

# HandLandmarkerResult(
#     handedness=[[Category(index=1, score=0.9865500926971436, display_name="Left", category_name="Left")]],
#     hand_landmarks=[
#         [
#             NormalizedLandmark(x=0.7697514295578003, y=0.6468967795372009, z=2.972396373479569e-07, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.668741762638092, y=0.6643492579460144, z=-0.040008749812841415, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.5648773312568665, y=0.6172918677330017, z=-0.06133072078227997, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.4947565197944641, y=0.5577317476272583, z=-0.08032052218914032, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.4326040744781494, y=0.526371419429779, z=-0.0983131155371666, visibility=0.0, presence=0.0),
#             NormalizedLandmark(
#                 x=0.5639973878860474, y=0.42474114894866943, z=-0.027148446068167686, visibility=0.0, presence=0.0
#             ),
#             NormalizedLandmark(x=0.5079708099365234, y=0.32789477705955505, z=-0.05173785239458084, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.480471134185791, y=0.2685859501361847, z=-0.07474029809236526, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.4576159119606018, y=0.21509180963039398, z=-0.09316738694906235, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.6137695908546448, y=0.3705744445323944, z=-0.030617333948612213, visibility=0.0, presence=0.0),
#             NormalizedLandmark(
#                 x=0.5645943880081177, y=0.23626628518104553, z=-0.049807462841272354, visibility=0.0, presence=0.0
#             ),
#             NormalizedLandmark(x=0.5348491668701172, y=0.1555137187242508, z=-0.06909414380788803, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.5079306960105896, y=0.08884336054325104, z=-0.08486079424619675, visibility=0.0, presence=0.0),
#             NormalizedLandmark(
#                 x=0.6730667352676392, y=0.34885501861572266, z=-0.040538471192121506, visibility=0.0, presence=0.0
#             ),
#             NormalizedLandmark(x=0.6328999996185303, y=0.22224289178848267, z=-0.06340891867876053, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.6061319708824158, y=0.14483919739723206, z=-0.08335975557565689, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.5819135904312134, y=0.07867544889450073, z=-0.09826647490262985, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.741492509841919, y=0.349530965089798, z=-0.05467082932591438, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.7340850830078125, y=0.24912577867507935, z=-0.07368527352809906, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.7289763689041138, y=0.18323004245758057, z=-0.08507121354341507, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.7216171026229858, y=0.12328104674816132, z=-0.09445909410715103, visibility=0.0, presence=0.0),
#         ]
#     ],
#     hand_world_landmarks=[
#         [
#             Landmark(x=0.046565815806388855, y=0.07342524081468582, z=0.006881324574351311, visibility=0.0, presence=0.0),
#             Landmark(x=0.010594870895147324, y=0.07021936029195786, z=-0.007981224916875362, visibility=0.0, presence=0.0),
#             Landmark(x=-0.014852593652904034, y=0.06191084906458855, z=-0.0166409183293581, visibility=0.0, presence=0.0),
#             Landmark(x=-0.045510780066251755, y=0.04683224856853485, z=-0.025354573503136635, visibility=0.0, presence=0.0),
#             Landmark(x=-0.06854020804166794, y=0.03167642652988434, z=-0.0283240657299757, visibility=0.0, presence=0.0),
#             Landmark(x=-0.028759997338056564, y=0.014417590573430061, z=0.0012775243958458304, visibility=0.0, presence=0.0),
#             Landmark(x=-0.04029626026749611, y=-0.010929192416369915, z=-0.008486682549118996, visibility=0.0, presence=0.0),
#             Landmark(x=-0.050002872943878174, y=-0.025899965316057205, z=-0.02284790761768818, visibility=0.0, presence=0.0),
#             Landmark(x=-0.059284575283527374, y=-0.0319778136909008, z=-0.05590397119522095, visibility=0.0, presence=0.0),
#             Landmark(x=-0.007565224077552557, y=-0.0017618824495002627, z=0.008078535087406635, visibility=0.0, presence=0.0),
#             Landmark(x=-0.021680276840925217, y=-0.038745105266571045, z=-0.006962615065276623, visibility=0.0, presence=0.0),
#             Landmark(x=-0.03542018681764603, y=-0.056135863065719604, z=-0.02756108157336712, visibility=0.0, presence=0.0),
#             Landmark(x=-0.047300662845373154, y=-0.06984193623065948, z=-0.05037245526909828, visibility=0.0, presence=0.0),
#             Landmark(x=0.01660585217177868, y=-0.0115079116076231, z=-0.0009205307578667998, visibility=0.0, presence=0.0),
#             Landmark(x=0.003100059926509857, y=-0.03883817046880722, z=-0.014047347940504551, visibility=0.0, presence=0.0),
#             Landmark(x=-0.008930228650569916, y=-0.05718749761581421, z=-0.033243678510189056, visibility=0.0, presence=0.0),
#             Landmark(x=-0.0204799622297287, y=-0.06876767426729202, z=-0.05420058220624924, visibility=0.0, presence=0.0),
#             Landmark(x=0.036402590572834015, y=-0.003861718811094761, z=-0.007347520440816879, visibility=0.0, presence=0.0),
#             Landmark(x=0.03788445517420769, y=-0.029805485159158707, z=-0.01170967984944582, visibility=0.0, presence=0.0),
#             Landmark(x=0.03136403486132622, y=-0.04963252693414688, z=-0.02345612645149231, visibility=0.0, presence=0.0),
#             Landmark(x=0.027734559029340744, y=-0.06272327154874802, z=-0.03760094568133354, visibility=0.0, presence=0.0),
#         ]
#     ],
# )

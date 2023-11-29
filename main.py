import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import logging, coloredlogs

from utils import draw_fingercount_on_image

logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG", logger=logger)  # logger 설정, logger.debug() 함수로 로그메시지 표시


def main():
    cap = cv2.VideoCapture(0)
    # TODO 핑거 카운터 알고리즘 설정
    # TODO 동영상 저장 설정

    # https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python#video
    base_options = python.BaseOptions(model_asset_path="models/hand_landmarker.task")
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    with vision.HandLandmarker.create_from_options(options) as detector:
        while True:
            success, image_origin = cap.read()

            if success:
                image = cv2.cvtColor(image_origin, cv2.COLOR_BGR2RGB)
                rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
                detection_result = detector.detect(rgb_frame)  # notebook/HandLandmarkerResult.ipynb 참조

                if detection_result:  # detection_result 로그데이터 생성
                    logger.debug(f"detection_result:\n {detection_result}")

                annotated_image = draw_fingercount_on_image(  # 원본 카메라 영상에 detection 결과를 합성
                    rgb_image=image_origin,
                    detection_result=detection_result,
                    # TODO 핑거 카운터 알고리즘 추가
                )

                cv2.imshow(f"Finger Counter", annotated_image)  # 동영상 재생
                # TODO 동영상 저장 코드 추가

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                logger.critical(f"cap.read() error")
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import logging, coloredlogs

import algorithm
from utils import extract_subclass, select_algorithm, get_args, draw_fingercount_on_image
from utils import get_video_writer, save_data

algos = extract_subclass(algorithm, algorithm.Algorithm)
args = get_args()  # get command-line argument to args

logger = logging.getLogger(__name__)  # set module name('__main__') to logger
coloredlogs.install(level=args.log.upper(), logger=logger)  # logger 설정


def main():
    cap = cv2.VideoCapture(0)
    algo = select_algorithm(algos, args)
    if args.save:  # 비디오 영상저장 객체 생성(out)
        out, filename = get_video_writer(algo)
        logger.info(f"video streaming will be saved to '{filename}' file")

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
                    algo=algo,
                )

                cv2.imshow(f"Finger Counter", annotated_image)  # 동영상 재생
                if args.save:  # save annotated_image to .avi file
                    out.write(annotated_image)

                key_pressed = cv2.waitKey(1) & 0xFF
                if key_pressed == ord("q"):  # 프로그램 종료
                    break
                elif key_pressed == ord("s"):  # 'results/data' 폴더로 이미지, 데이터 저장
                    save_data(detection_result.hand_landmarks, annotated_image)

            else:
                logger.error(f"cap.read() error")
                break

    cap.release()
    if args.save:
        out.release()
        logger.info(f"video streaming saved to '{filename}' file")
    cv2.destroyAllWindows()
    logger.info(f"program exit normally\n")


if __name__ == "__main__":
    main()

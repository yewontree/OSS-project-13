import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

import logging, coloredlogs

import algorithm
from utils import extract_subclass, select_algorithm, get_args, draw_fingercount_on_image
from utils import ms_timestamp, get_video_writer, to_datetime_str, save_data

algos = extract_subclass(algorithm, algorithm.Algorithm)
args = get_args()

logger = logging.getLogger(__name__)
coloredlogs.install(level=args.log.upper(), logger=logger)  # logger 설정


def main():
    cap = cv2.VideoCapture(0)
    algo = select_algorithm(algos, args)
    if args.save:  # 동영상 저장 설정
        out, filename = get_video_writer(algo)
        logger.info(f"video streaming will be saved to '{filename}'")

    detection_result = None  # 콜백(callback)함수 리턴값 저장용 변수 (callback 함수내에서 nonlocal로 access한다)
    detection_image = None
    detection_time = None

    # Create a hand landmarker instance with the live stream mode:
    def callback_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        nonlocal detection_result, detection_image, detection_time
        detection_result = result
        detection_image = output_image.numpy_view()
        detection_time = timestamp_ms
        logger.debug(f"hand landmarker result: timestamp_ms: {timestamp_ms}\n {result}")

    # https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python#video
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=callback_result,  # async detector 객체가 작업완료시 이 함수를 호출한다.
        num_hands=2,
    )
    with HandLandmarker.create_from_options(options) as detector:
        while True:
            success, image_origin = cap.read()

            if success:
                # image = cv2.cvtColor(image_origin, cv2.COLOR_BGR2RGB)
                rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_origin)
                detector.detect_async(rgb_frame, ms_timestamp())  # notebook/HandLandmarkerResult.ipynb 참조

                if detection_result:
                    annotated_image = draw_fingercount_on_image(  # 원본 카메라 영상에 detection 결과를 합성
                        rgb_image=detection_image,
                        detection_result=detection_result,
                        algo=algo,
                        timestamp=to_datetime_str(detection_time // 1000),
                    )
                else:
                    annotated_image = image_origin

                cv2.imshow(f"Finger Counter", annotated_image)  # 동영상 재생
                if args.save:  # 동영상 저장
                    out.write(annotated_image)

                key_pressed = cv2.waitKey(1) & 0xFF
                if key_pressed == ord("q"):
                    break
                elif key_pressed == ord("s"):
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

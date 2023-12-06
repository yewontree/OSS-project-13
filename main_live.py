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
from utils import ms_timestamp_now, get_video_writer, to_datetime_str, save_data
from utils import DetectInfo

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

    d = DetectInfo()  # for callback processing object

    # Create a hand landmarker instance with the live stream mode:
    def callback_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        nonlocal d
        d.update_live(result, output_image, timestamp_ms)
        logger.debug(f"hand landmarker result: timestamp_ms: {d.time_diff}, avg: {d.time_diff_mean:4.1f}\n {result}")

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
                detector.detect_async(rgb_frame, ms_timestamp_now())  # notebook/HandLandmarkerResult.ipynb 참조

                if d.new_detection():
                    annotated_image = draw_fingercount_on_image(  # 원본 카메라 영상에 detection 결과를 합성
                        rgb_image=d.detection_image,
                        detection_result=d.detection_result,
                        algo=algo,
                        timestamp=to_datetime_str(d.detection_time // 1000),
                    )

                    cv2.imshow(f"Finger Counter", annotated_image)  # 동영상 재생
                    if args.save:  # 동영상 저장
                        out.write(annotated_image)

                    d.reset_detection()

                key_pressed = cv2.waitKey(1) & 0xFF
                if key_pressed == ord("q"):
                    break
                elif key_pressed == ord("s"):
                    save_data(d.detection_result.hand_landmarks, annotated_image)

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

import os
import logging, coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO", logger=logger)  # logger 설정, logger.info() 함수로 로그메시지 표시

root: str = os.path.dirname(__file__)  # 이 파일(settings.py)이 속한 폴더를 root명으로 추출

# 미디어파이프(mediapipe) 핸드 랜드마크 검출 예제에서 사용하는 파라미터
MARGIN = 10  # pixels 글자를 표시할 이미지 여백
FONT_SIZE = 1  # 영상 이미지에 표시하는 문자 사이즈
FONT_THICKNESS = 2  # 영상 이미지에 표시하는 문자 폰트 두께
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green, 문자의 색 지정

# 미디어파이프 모듈에서 사용하는 핸드랜드마커 모델의 인덱스(0~20 총 21개) 가운데 손가락 끝을 가리키는 인덱스(index)
finger_tips = [4, 8, 12, 16, 20]

# TODO 글로벌 변수 또는 상수가 필요할 경우 여기에 설정.
# ...


logger.info(f"importing {os.path.basename(__file__)}")
logger.info(f"root 폴더: {root}")
logger.info(f"finger_tips index: {finger_tips}")

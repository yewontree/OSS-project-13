# Finger Counter using OpenCV and MediaPipe
## 1. 개요
 - MediaPipe 핸드 데이터로부터 핑거 카운터 구현
 - Counting 알고리즘과 프로그램 코드를 분리하여 알고리즘 개선에 유리하도록 함


## 2. 가상환경 생성(Virtual Environment Setup)
First install miniconda, and execute following command
``` bash
conda create -n oss python=3.11 -y
conda activate oss
pip install jupyterlab  --> jupyter lab 설치
pip install notebook    --> jupyter notebook 설치
pip install opencv-python
pip install mediapipe
pip install coloredlogs

```

## 3. OSS-project-13 클론(clone) & execute jupyter notebook 실행
``` bash
git clone ghttps://github.com/yewontree/OSS-project-13.git
cd ./OSS-project-13 --> 폴더 이동
jupyter notebook

```


## 4. 프로그램 실행
```bash
❯ python main.py --help
usage: main.py [-h] [--algo {john,jane}] [--log {debug,info,warning,error,critical}] [--log_a {debug,info,warning,error,critical}] [--save]

options:
  -h, --help            show this help message and exit
  --algo {john,jane}
                        select finger counter algorithm
  --log {debug,info,warning,error,critical}
                        select log level
  --log_a {debug,info,warning,error,critical}
                        select algorithm log level
  --save                save video data as .avi file


❯ python main.py --algo john
❯ python main.py --algo jane
❯ python main.py --algo jane --log dubug
❯ python main.py --algo jane --log dubug --save
...
```

## 5. 참조 사이트
 - https://github.com/google/mediapipe/blob/master/docs/solutions/hands.md
 - https://github.com/GasbaouiMohammedAlAmin/Finger-Counter-using-Hand-Tracking-And-Open-cv.git
 - https://youtu.be/01sAkU_NvOY?si=2X6JEZ-9QbW9Dhpy
 - https://github.com/topics/finger-count-recognition

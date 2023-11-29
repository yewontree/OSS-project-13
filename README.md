# Finger Counter using OpenCV and mediapipe
## 1. 개요
 - Mediapipe 핸드 데이터로부터 숫자 인식
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
cd ~/OSS-project-13 --> 폴더 이동
jupyter notebook

```


## 4. 프로그램 실행
```bash
python main.py
```

## 5. 참조 사이트
 - https://github.com/GasbaouiMohammedAlAmin/Finger-Counter-using-Hand-Tracking-And-Open-cv.git
 - https://github.com/topics/finger-count-recognition

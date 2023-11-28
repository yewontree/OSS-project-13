# OSS-project-13

## 1. 가상환경 생성(Virtual Environment Setup)
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

## 2. OSS-project-13 클론(clone) & execute jupyter notebook 실행
``` bash
git clone ghttps://github.com/yewontree/OSS-project-13.git
cd ~/OSS-project-13 --> 폴더 이동
jupyter notebook 

```
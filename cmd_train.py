import os
from pathlib import Path
import torch

# YOLOv5 라이브러리 경로
# yolov5_path = Path('C:/Users/GODJUHYEOK/Desktop/findog/yolov5')  # YOLOv5 디렉토리 경로

# 데이터셋 구성 파일 경로
data_yaml = Path('data.yaml')

# 모델 아키텍처 설정 (yolov5s, yolov5m, yolov5l, yolov5x 등)
model_architecture = 'yolov5m'

# 하이퍼파라미터 설정
epochs = 100
batch_size = 16
img_size = 640

# 학습 명령어 구성
train_command = f"""
python train.py --img {img_size} --batch {batch_size} --epochs {epochs} --data {data_yaml} --cfg {model_architecture}.yaml --weights {model_architecture}.pt
"""

# YOLOv5 디렉토리로 이동하여 학습 실행
os.system(train_command)

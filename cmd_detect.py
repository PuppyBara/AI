import os
from pathlib import Path
import torch


# detect 해보고 Yolo 모델 경로
yolo_path = Path('./runs/train/exp4/weights/best.pt')

# detect 해보고 싶은 이미지 파일 / 폴더 경로
img_path = Path('./data/images/bong.jpg')

# 학습 명령어 구성
detect_command = f"""
python crop_nose.py --weights {yolo_path} --img 718 --conf 0.25 --source {img_path} --save-crop
"""

# YOLOv5 디렉토리로 이동하여 학습 실행
os.system(detect_command)

import uvicorn
from fastapi import FastAPI, UploadFile, Request, status, File
import os
from pathlib import Path
import sys
import platform
import os
import dotenv
import boto3
import PIL.Image as Image
from pydantic import BaseModel
from ultralytics import YOLO
from ultralytics.engine.results import Results
model = YOLO("./runs/detect/train2/weights/best.pt")

# Pydantic 모델 정의
class Info(BaseModel):
    lostDogInfo: dict
    images: list
    upperBound: float

# from predict import run;
app = FastAPI()

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
print("Hello")


# .env 파일 로드
dotenv.load_dotenv()

# AWS S3 setting
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION_NAME = os.getenv('AWS_REGION_NAME')
AWS_BUCKET_NAME = os.getenv('AWS_BUCKET_NAME')
s3 = boto3.resource('s3',
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key= AWS_SECRET_ACCESS_KEY,
                    region_name=AWS_REGION_NAME)
BUCKET = s3.Bucket(AWS_BUCKET_NAME)


def s3_load_image(img_url):
  return BUCKET.Object(img_url).get()['Body']


@app.post("/ai/analyse")
def analyse(find_info: Info):
    # S3에 저장된 이미지 파일명이 img.png일 때
    # find_info = Request.body["find_info"]
    lost_dog = find_info.lostDogInfo
    dog_candidates = find_info.images

    lost_img = s3_load_image(lost_dog["s3Url"])
    lost_img = Image.open(lost_img)
    # lost_img.show()
    candidate_images = [Image.open(s3_load_image(dog["s3Url"])) for dog in dog_candidates]
    
    lost_pred = model(lost_img)
    pred = model(candidate_images)

    for idx, temp in enumerate(pred):

        cropped_image = crop_nose(candidate_images[idx], temp)

        if temp.boxes.conf > 0.7:
            # 개 코 인지 되면 비문 인식 AI 적용
            nose_detection(cropped_image)
        else:
            # 개 코 인지 안됐으므로 안면 인식 AI 적용
            face_detection(cropped_image)

    return "test" 


def crop_nose(image: Image, result: Results):
    # 자를 좌표 (left, upper, right, lower)

    crop_box = tuple(result.boxes.xyxy[0].T.tolist())
    # 이미지 자르기
    return image.crop(crop_box)


def nose_detection(image):
    print(f"nose detection call - {type(image)}")

def face_detection(image):
    print(f"face detection call - {type(image)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
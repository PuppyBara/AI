import uvicorn
from fastapi import FastAPI, UploadFile, Request, status, File
from fastapi.responses import JSONResponse, FileResponse
import os
import uuid
from pathlib import Path
import sys
import platform
import json
import os
import dotenv
import boto3
import PIL.Image as Image
from pydantic import BaseModel
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

# S3에 저장된 이미지 파일명이 img.png일 때
# img = s3_load_image('img.png')
# img = Image.open(img)
# img.show()

# from starlette.middleware.cors import CORSMiddleware

# origins = [
#     "https://j10b102.p.ssafy.io",
#     "http://localhost:8000",
#     "http://0.0.0.0:8000",
#     "http://0.0.0.0",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

@app.post("/ai/analyse")
def test(find_info: Info):
    # S3에 저장된 이미지 파일명이 img.png일 때
    # find_info = Request.body["find_info"]
    print("!")
    print(find_info)
    lost_dog = find_info.lostDogInfo
    dog_candidates = find_info.images

    lost_img = s3_load_image(lost_dog["s3Url"])
    lost_img = Image.open(lost_img)
    # lost_img.show()
    candidate_images = [Image.open(s3_load_image(dog["s3Url"])) for dog in dog_candidates]

    return "test"


# @app.post("/python/upload")
# async def upload_file(image: UploadFile = File(...)):
#     extension = os.path.splitext(image.filename)[1]
#     unique_name = f"{uuid.uuid4()}{extension}"

#     upload_path = "uploads"
#     if not os.path.exists(upload_path):
#         os.makedirs(upload_path)
#     file_path = os.path.join(upload_path, unique_name)
#     with open(file_path, "wb") as buffer:
#         buffer.write(image.file.read())
    
#     option = {
#         "weights":ROOT / './best.pt',
#         "source" : file_path,
#         "data": ROOT/'data/custom.yaml',
#         "imgsz" : [640,640],
#         "device" : "cpu"
#     }


#     # option = {
#     #     "weights":ROOT / './trained_model/2515_2659_2066_1224_2558_109.pt',
#     #     "source" : "https://yoyak.s3.ap-northeast-2.amazonaws.com/1.jpg",
#     #     "data": ROOT/'data/custom.yaml',
#     #     "imgsz" : [640,640],
#     #     "device" : "cpu"
#     # }
#     print("option", option)
#     names = run(**option)

#     medicineList = []
#     for name in names:
#         data = {}
#         splited = name.split("-")
#         data["medicineCode"] = splited[0]
#         data["medicineName"] = splited[1]
#         medicineList.append(data)

    


#     return {
#         "count": len(names),
#         "medicineList": medicineList
#     }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
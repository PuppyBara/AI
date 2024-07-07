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
import numpy as np
from pydantic import BaseModel
from ultralytics import YOLO
from ultralytics.engine.results import Results
from imgaug import augmenters as iaa
from tensorflow.keras.models import load_model

# 아마 서버에 dlib가 깔려야 할꺼에요...
# 그래서 그게 해결돼야 이 및의 부분이 실행이 될겁니다.
import dlib
import imutils
from imutils import face_utils
import face_recognition

model = YOLO("./runs/detect/train2/weights/best.pt")

# 비문 인식 모델 로드
model_nose = load_model('./models/original.h5')

# 안면 인식 모델 로드
detector = dlib.cnn_face_detection_model_v1('./models/dlibModels/dogHeadDetector.dat')
predictor = dlib.shape_predictor('./models/dlibModels/landmarkDetector.dat')

known_face_encodings = []
known_face_names = []

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
    upper_bound = find_info.upperBound

    lost_img = s3_load_image(lost_dog["s3Url"])
    lost_img = Image.open(lost_img)
    # lost_img.show()
    # candidate_images = [Image.open(s3_load_image(dog["s3Url"])) for dog in dog_candidates]
    candidate_images = [(dog.dogId, Image.open(s3_load_image(dog.s3Url))) for dog in dog_candidates]
    
    ai_results = []
    
    lost_pred = model(lost_img)
    # pred = model(candidate_images)
    pred = model([img for _, img in candidate_images])

    for idx, temp in enumerate(pred):
        dog_id = dog_candidates[idx].dogId
        # cropped_image = crop_nose(candidate_images[idx][1], temp)
        if temp.boxes.conf > 0.7:
            # 개 코 인지 되면 비문 인식 AI 적용
            # nose_detection(cropped_image)
            cropped_image = crop_nose(candidate_images[idx][1], temp)
            detection_result = nose_detection(lost_img, cropped_image, upper_bound)
            detection_type = "Nose"
        else:
            # 개 코 인지 안됐으므로 안면 인식 AI 적용
            # face_detection(cropped_image)
            detection_results = face_detection(lost_img, candidate_images, upper_bound)
            for result in detection_results:
                ai_results.append({
                    "dogId": result[0],
                    "percentage": result[1],
                    "aiType": result[2]
                })
        if detection_result:
            ai_results.append({
                "dogId": dog_id,
                "percentage": detection_result,
                "aiType": detection_type
            })
    # return "test" 
    return {"aiDogResult": ai_results}

def crop_nose(image: Image, result: Results):
    # 자를 좌표 (left, upper, right, lower)

    crop_box = tuple(result.boxes.xyxy[0].T.tolist())
    # 이미지 자르기
    return image.crop(crop_box)

def preprocess_image(image: Image.Image):
    # 이미지 흑백 변환 및 크기 조정
    image = np.array(image.convert('L').resize((96, 96)))
    image = image.astype(np.float32) / 255.0
    image = image.reshape((1, 96, 96, 1))
    
    # 이미지 증강
    seq = iaa.Sequential([
        iaa.GaussianBlur(sigma=(0, 0.5)),
        iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-30, 30),
            order=[0, 1],
            cval=255
        )
    ], random_order=True)
    
    # 증강된 이미지 반환
    augmented_image = seq.augment_image(image)
    return augmented_image.reshape((1, 96, 96, 1))

def nose_detection(lost_img, cropped_img, upper_bound):
    preprocessed_lost_img = preprocess_image(lost_img)
    preprocessed_candidate_img = preprocess_image(cropped_img)
    
    pred = model_nose.predict([preprocessed_lost_img, preprocessed_candidate_img])
    pred_confidence = pred[0][0]*100
    
    if pred_confidence >= upper_bound:
        return pred_confidence
    return None

def face_detection(lost_img, candidate_images, upper_bound):
    known_face_encodings.clear()
    known_face_names.clear()
    results = []

    for dog_id, img in candidate_images:
        add_known_face(img, str(dog_id))

    face_names, face_percentages = name_labeling(lost_img)

    for name, percentage in zip(face_names, face_percentages):
        dog_id = int(name)
        if (percentage + 30) >= upper_bound:
            results.append((dog_id, percentage + 30, "Face"))

    return results

# face_recognition이 강아지 얼굴을 인식하도록 하는 함수

def _trim_css_to_bounds(css, image_shape):
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)


def _rect_to_css(rect):
    return rect.top(), rect.right(), rect.bottom(), rect.left()


def _raw_face_locations(img, number_of_times_to_upsample=1):
    return detector(img, number_of_times_to_upsample)


def face_locations(img, number_of_times_to_upsample=1):
    return [_trim_css_to_bounds(_rect_to_css(face.rect), img.shape) for face in _raw_face_locations(img, number_of_times_to_upsample)]

def add_known_face(face_image, name):
    dets_locations = face_locations(face_image, 1)
    face_encoding = face_recognition.face_encodings(face_image, dets_locations)[0]

    known_face_encodings.append(face_encoding)
    known_face_names.append(name)
    
def name_labeling(input_image, size=None):
    image = input_image.copy()

    if size:
        image = imutils.resize(image, width=size)

    dets_locations = face_locations(image)
    face_encodings = face_recognition.face_encodings(image, dets_locations)

    face_names = []
    face_percentages = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
        name = "0"
        percentage = 0.0

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        percentage = (1 - face_distances[best_match_index]) * 100
        face_names.append(name)
        face_percentages.append(percentage)

    return face_names, face_percentages

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
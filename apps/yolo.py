from pathlib import Path
from typing import List
from PIL import Image, ImageFilter
from ultralytics import YOLO
import numpy as np

model_path = Path("./apps/yolov11n-face.pt")


class YOLOModel:
    def __init__(self):
        self._model = YOLO(model_path.absolute(), task="detect", verbose=False)
        pass

    def detect_faces(self, image_path: str, conf: float, iou: float) -> List[tuple[int, ...]] | None:
        results = self._model.predict(image_path, conf=conf, iou=iou, verbose=False)

        has_faces = any(result.boxes for result in results)

        if not has_faces:
            return None

        face_boxes = [tuple(map(int, box.xyxy[0])) for result in results for box in result.boxes]

        return face_boxes

def blur_faces(face_boxes: List[tuple[int, ...]], image: Image, radius: int = 20) -> Image:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    for face_box in face_boxes:
        region = image.crop(face_box).filter(ImageFilter.GaussianBlur(radius))
        image.paste(region, face_box)
    return image
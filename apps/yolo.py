from pathlib import Path
from typing import List
from PIL import Image, ImageFilter, ImageDraw
from ultralytics import YOLO
import numpy as np

#Ämodel_path = Path("./apps/yolov11n-face.pt")
model_path = Path("./apps/yolov11n-face_openvino_model" )

class YOLOModel:
    def __init__(self):
        self._model = YOLO(model_path.absolute(), task="detect", verbose=False)
        pass

    def detect_faces(self, image_path: str, conf: float, iou: float) -> List[tuple[int, ...]] | None:
        results = self._model.track(image_path, conf=conf, iou=iou, verbose=False)

        has_faces = any(result.boxes for result in results)

        if not has_faces:
            return None

        face_boxes = [tuple(map(int, box.xyxy[0])) for result in results for box in result.boxes]

        return face_boxes

def blur_faces(face_boxes: List[tuple[int, ...]], image: np.ndarray, radius: int = 20) -> np.ndarray:
    pil_image = Image.fromarray(image)
    for face_box in face_boxes:
        # Agrandir légèrement la boîte
        x0, y0, x1, y1 = face_box
        x0, y0, x1, y1 = x0 - 10, y0 - 10, x1 + 10, y1 + 10
        
        # Créer un masque pour l'ellipse
        mask = Image.new('L', pil_image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse([x0, y0, x1, y1], fill=255)
        
        # Appliquer le flou gaussien à la région elliptique
        blurred_image = pil_image.filter(ImageFilter.GaussianBlur(radius))
        pil_image.paste(blurred_image, mask=mask)
    return np.array(pil_image)

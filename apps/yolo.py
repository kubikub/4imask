from pathlib import Path
from typing import List
from PIL import Image, ImageFilter, ImageDraw
from ultralytics import YOLO
import numpy as np
from typing import Tuple
import skimage.draw
# from torch import cuda
import cv2
from apps.utils import resource_path
#model_path = Path("./apps/yolov11n-face.pt")
# model_path = Path("./apps/yolov11n-face_openvino_model" )
import GPUtil as gpu
class YOLOModel:
    def __init__(self):
        # Check if GPU is available and has more than 0 GPUs
        print(gpu.getAvailable())
        if gpu.getAvailable() and gpu.getGPUs() > 0:
            self.device = "cuda"
            self.model_path = resource_path("res/models/yolov11n-face.pt")
        else:
            self.device = None
            self.model_path = resource_path("res/models/yolov11n-face_openvino_model/" )
            print("Using openvino for YOLO model")
        
        self._model = YOLO(self.model_path, task="detect", verbose=False)
        pass

    def detect_faces(self, image_path: str, conf: float, iou: float, max_width: int = None, max_height: int = None) -> List[tuple[int, ...]] | None:
        
        results = self._model.predict(image_path, conf=conf, iou=iou, verbose=False, device=self.device, half=True, agnostic_nms=True)

        has_faces = any(result.boxes for result in results)

        if not has_faces:
            return None

        face_boxes = []
        for result in results:
            for box in result.boxes:
                x0, y0, x1, y1 = map(int, box.xyxy[0])
                if (max_width is None or (x1 - x0) <= max_width) and (max_height is None or (y1 - y0) <= max_height):
                    face_boxes.append((x0, y0, x1, y1))

        return face_boxes


    def mask_faces(self, face_boxes: List[tuple[int, ...]], frame: np.ndarray, radius: int = 30,
                replacewith: str = 'blur', mask_size : float = 1.3, ellipse = True, mosaicsize: int = 15) -> np.ndarray:
        for face_box in face_boxes:
            # Agrandir légèrement la boîte
            x1, y1, x2, y2 = face_box
            x1, y1, x2, y2 = self.scale_bb(x1, y1, x2, y2, mask_size)
            # Clip bb coordinates to valid frame region
            y1, y2 = max(0, y1), min(frame.shape[0] - 1, y2)
            x1, x2 = max(0, x1), min(frame.shape[1] - 1, x2)
            if replacewith == 'solid':
                ovcolor = (0, 0, 0)  # Couleur noire pour masquer
                cv2.rectangle(frame, (x1, y1), (x2, y2), ovcolor, -1)
            elif replacewith == 'blur':
                bf = 2  # blur factor (number of pixels in each dimension that the face will be reduced to)
                blurred_box = cv2.blur(
                    frame[y1:y2, x1:x2],
                    (abs(x2 - x1) // bf, abs(y2 - y1) // bf)
                )
                if ellipse:
                    roibox = frame[y1:y2, x1:x2]
                    # Get y and x coordinate lists of the "bounding ellipse"
                    ey, ex = skimage.draw.ellipse((y2 - y1) // 2, (x2 - x1) // 2, (y2 - y1) // 2, (x2 - x1) // 2)
                    roibox[ey, ex] = blurred_box[ey, ex]
                    frame[y1:y2, x1:x2] = roibox
                else:
                    frame[y1:y2, x1:x2] = blurred_box
            elif replacewith == 'mosaic':
                for y in range(y1, y2, mosaicsize):
                    for x in range(x1, x2, mosaicsize):
                        pt1 = (x, y)
                        pt2 = (min(x2, x + mosaicsize - 1), min(y2, y + mosaicsize - 1))
                        color = (int(frame[y, x][0]), int(frame[y, x][1]), int(frame[y, x][2]))
                        cv2.rectangle(frame, pt1, pt2, color, -1)
            return frame

    def scale_bb(self, x0, y0, x1, y1, mask_scale=1.3):
        s = mask_scale - 1.0
        h, w = y1 - y0, x1 - x0
        y0 -= h * s
        y1 += h * s
        x0 -= w * s
        x1 += w * s
        return np.round([x0, y0, x1, y1]).astype(int)

from typing import List
from ultralytics import YOLO
import numpy as np
from torch.cuda import (is_available as cuda_is_available,
                        device_count as cuda_device_count)

from apps.utils import resource_path, draw_detections
import logging

class YOLOModel:
    def __init__(self):
        # Check if GPU is available and has more than 0 GPUs
        self.logger = logging.getLogger(__name__).getChild(self.__class__.__name__)
        if cuda_is_available() and cuda_device_count() > 0:
            self.device = "cuda"
            self.model_path = resource_path("res/models/yolov11n-face.pt")
            self.logger.info("Using CUDA for YOLO model")
        else:
            self.device = None
            self.model_path = resource_path("res/models/yolov11n-face_openvino_model/" )
            self.logger.info("Using openvino for YOLO model")
        
        self._model = YOLO(self.model_path, task="detect", verbose=False)
        pass

    def detect_faces(self, image_path: str, conf: float, iou: float, max_width: int = None, max_height: int = None) -> List[tuple[int, ...]] | None:
        
        results = self._model.predict(image_path, conf=conf, iou=iou,
                                    verbose=False, device=self.device,
                                    half=True, agnostic_nms=True)

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
            frame = draw_detections(frame, x1, y1, x2, y2, replacewith, mask_size, ellipse, mosaicsize)
        return frame
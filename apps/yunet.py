# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

from typing import Tuple
import numpy as np
import cv2 as cv2
from apps.utils import resource_path, draw_detections
import logging


class YuNet:
    def __init__(self, inputSize=[320, 320], confThreshold=0.6, nmsThreshold=0.3, topK=5000, backendId=0, targetId=0):
        self.logger = logging.getLogger(self.__class__.__name__).getChild(self.__class__.__name__)
        self._modelPath = resource_path("res/models/face_detection_yunet_2023mar.onnx")
        self._inputSize = tuple(inputSize)  # [w, h]
        self._confThreshold = confThreshold
        self._nmsThreshold = nmsThreshold
        self._topK = topK
        self._backendId = backendId
        self._targetId = targetId

        self._model = cv2.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=self._topK,
            backend_id=self._backendId,
            target_id=self._targetId)

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backendId, targetId):
        self._backendId = backendId
        self._targetId = targetId
        self._model = cv2.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=self._topK,
            backend_id=self._backendId,
            target_id=self._targetId)

    def setInputSize(self, input_size):
        self._model.setInputSize(tuple(input_size))

    def infer(self, image):
        # Forward
        faces = self._model.detect(image)
        return np.empty(shape=(0, 5)) if faces[1] is None else faces[1]

    def visualize(
            self, image, results, mask_scale=1.3, replacewith='blur', ellipse: bool = True,
            draw_scores: bool = False,
            ovcolor: Tuple[int] = (0, 0, 0),
            replaceimg=None,
            mosaicsize: int = 2,
            box_color=(0, 255, 0),
            text_color=(0, 0, 255), fps=None
            ):
        output = image.copy()
        # landmark_color = [
        #     (255,   0,   0),  # right eye
        #     (0,   0, 255),  # left eye
        #     (0, 255,   0),  # nose tip
        #     (255,   0, 255),  # right mouth corner
        #     (0, 255, 255)  # left mouth corner
        # ]

        if fps is not None:
            cv2.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

        for det in results:
            bbox = det[0:4].astype(np.int32)
            # x2, y2, x1, y1 = bbox
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
            output = draw_detections(output, x1, y1, x2, y2, replacewith, mask_scale, ellipse, mosaicsize)
        return output

    def scale_bb(self, x1, y1, x2, y2, mask_scale=1.0):
        s = mask_scale - 1.0
        h, w = y2 - y1, x2 - x1
        y1 -= h * s
        y2 += h * s
        x1 -= w * s
        x2 += w * s
        return np.round([x1, y1, x2, y2]).astype(int)

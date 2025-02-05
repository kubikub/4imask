# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

from itertools import product
from typing import Tuple
import skimage.draw
import numpy as np
import cv2 as cv2
from apps.utils import resource_path

class YuNet:

    def __init__(self, inputSize=[320, 320], confThreshold=0.6, nmsThreshold=0.3, topK=5000, backendId=0, targetId=0):
        self._modelPath = resource_path("res/models/face_detection_yunet_2023mar.onnx")
        self._inputSize = tuple(inputSize) # [w, h]
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

    
    def visualize(self, image, results, mask_scale=1.3, replacewith='blur', ellipse: bool = True,
                  draw_scores: bool = False,
                  ovcolor: Tuple[int] = (0, 0, 0),
                  replaceimg=None,
                  mosaicsize: int = 2, 
                  box_color=(0, 255, 0), 
                  text_color=(0, 0, 255), fps=None):
        output = image.copy()
        landmark_color = [
            (255,   0,   0),  # right eye
            (0,   0, 255),  # left eye
            (0, 255,   0),  # nose tip
            (255,   0, 255),  # right mouth corner
            (0, 255, 255)  # left mouth corner
        ]

        if fps is not None:
            cv2.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

        for det in results:
            bbox = det[0:4].astype(np.int32)
            # x2, y2, x1, y1 = bbox
            x1, y1, x2, y2 =bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
            x1, y1, x2, y2 = self.scale_bb(x1, y1, x2, y2, mask_scale)
            # Clip bb coordinates to valid frame region
            y1, y2 = max(0, y1), min(output.shape[0] - 1, y2)
            x1, x2 = max(0, x1), min(output.shape[1] - 1, x2)
            if replacewith == 'solid':
                cv2.rectangle(output, (x1, y1), (x2, y2), ovcolor, -1)
            elif replacewith == 'blur':
                bf = 2  # blur factor (number of pixels in each dimension that the face will be reduced to)
                blurred_box = cv2.blur(
                    output[y1:y2, x1:x2],
                    (abs(x2 - x1) // bf, abs(y2 - y1) // bf)
                )
                if ellipse:
                    roibox = output[y1:y2, x1:x2]
                    # Get y and x coordinate lists of the "bounding ellipse"
                    ey, ex = skimage.draw.ellipse((y2 - y1) // 2, (x2 - x1) // 2, (y2 - y1) // 2, (x2 - x1) // 2)
                    roibox[ey, ex] = blurred_box[ey, ex]
                    output[y1:y2, x1:x2] = roibox
                else:
                    output[y1:y2, x1:x2] = blurred_box
            elif replacewith == 'img':
                target_size = (x2 - x1, y2 - y1)
                resized_replaceimg = cv2.resize(replaceimg, target_size)
                if replaceimg.shape[2] == 3:  # RGB
                    output[y1:y2, x1:x2] = resized_replaceimg
                elif replaceimg.shape[2] == 4:  # RGBA
                    output[y1:y2, x1:x2] = output[y1:y2, x1:x2] * (1 - resized_replaceimg[:, :, 3:] / 255) + resized_replaceimg[:, :, :3] * (resized_replaceimg[:, :, 3:] / 255)
            elif replacewith == 'mosaic':
                for y in range(y1, y2, mosaicsize):
                    for x in range(x1, x2, mosaicsize):
                        pt1 = (x, y)
                        pt2 = (min(x2, x + mosaicsize - 1), min(y2, y + mosaicsize - 1))
                        color = (int(output[y, x][0]), int(output[y, x][1]), int(output[y, x][2]))
                        cv2.rectangle(output, pt1, pt2, color, -1)
            elif replacewith == 'none':
                pass
            # ovcolor = (0, 0, 0)
            # cv2.rectangle(output, (x1, y1), (x2, y2), ovcolor, -1)
            # face = output[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            # anonymize
            # face = self.anonymize_face_pixelate(face)
            # output[bbox[1]:bbox[3], bbox[0]:bbox[2]] = face
            # cv2.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)

            # conf = det[-1]
            # cv2.putText(output, '{:.4f}'.format(conf), (bbox[0], bbox[1]+12), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color)

            # landmarks = det[4:14].astype(np.int32).reshape((5, 2))
            # for idx, landmark in enumerate(landmarks):
            #     cv2.circle(output, landmark, 2, landmark_color[idx], 2)

        return output

    def scale_bb(self, x1, y1, x2, y2, mask_scale=1.0):
        s = mask_scale - 1.0
        h, w = y2 - y1, x2 - x1
        y1 -= h * s
        y2 += h * s
        x1 -= w * s
        x2 += w * s
        return np.round([x1, y1, x2, y2]).astype(int)
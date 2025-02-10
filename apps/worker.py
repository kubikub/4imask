
import cv2
import time
from PySide6.QtWidgets import (QMessageBox)
from PySide6.QtCore import  QThread, Signal
import numpy as np
import imutils
import platform
from apps.yunet import YuNet
from apps.centerface import CenterFace
from apps.yolo import YOLOModel
import logging

class AnonymizationWorker(QThread):
    progress_updated = Signal(int)
    time_remaining_updated = Signal(int)
    anonymization_complete = Signal()
    frame_emited = Signal(np.ndarray)
    def __init__(self, video_path, output_format, mask_size, write_output=False):
        super().__init__()
        self.logger = logging.getLogger(__name__).getChild(self.__class__.__name__)
        self.video_path = video_path
        # self.net = net
        self.output_format = output_format
        self._is_paused = False
        self._is_stopped = False
        self.replacewith = 'blur'  # Default value
        self.mask_size = mask_size
        self.model = 'yolo'
        self.write_output = write_output
        self.logger.info("worker initialized")

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Error", "Failed to open video file.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        new_width, new_height = self.get_dimensions(self.output_format)
        self.logger.info(self.output_format)
        if self.write_output:
            if platform.system() == "Windows":
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # output_name = os.path.basename(self.video_path).split(".")[0] + "_anonymized_" + self.output_format.lower() + ".mkv"
            output_name = (self.video_path).split(".")[0] + "_anonymized_" + self.output_format.lower() + ".mp4"

            self.logger.info(output_name)
            out = cv2.VideoWriter(output_name, fourcc, fps, (new_width, new_height))
        # out = cv2.VideoWriter('/home/frank-kubler/anonymized_video2.avi', fourcc, fps)
            if not out.isOpened():
                QMessageBox.critical(self, "Error", "Failed to open video writer.")
                cap.release()
                return
        # centerface = CenterFace(in_shape=(new_width, new_height), backend='auto')  # auto
        if self.model == 'yolo':
            yolo_ = YOLOModel()
            centerface = None
            self.yunet = None
        elif self.model == 'centerface':
            centerface = CenterFace(in_shape=(new_width, new_height), backend='auto')  # auto
            yolo_ = None
            self.yunet = None
        elif self.model == 'yunet':
            self.yunet = YuNet([new_width, new_height])  # auto
            yolo_ = None
            centerface = None
        else:
            yolo_ = None
            centerface = None
            self.yunet = None
        self.logger.info(f'Anonymizing video with {self.model} model, mask size: {self.mask_size}, replace with: {self.replacewith}')
        # centerface.backend = 'onnxruntime-directml'
        start_time = time.time()
        frame_count = 0
        while cap.isOpened():
            if self._is_stopped:
                break
            if self._is_paused:
                time.sleep(0.1)
                continue
            ret, frame = cap.read()
            if not ret:
                break
            frame = imutils.resize(frame, width=new_width)
            if yolo_ is not None:
                faces = yolo_.detect_faces(frame, 0.25, 0.45)
                if faces is not None:
                    frame = yolo_.mask_faces(faces, frame, 20, self.replacewith, self.mask_size)
            elif centerface is not None:
                detections, _ = centerface(frame, threshold=0.25)
                centerface.anonymize_frame(detections, frame, self.mask_size, replacewith=self.replacewith, ellipse=True, draw_scores=False, replaceimg=None, mosaicsize=15)
            elif self.yunet is not None:
                frame = self.yunet_mechanism(frame, self.mask_size, replacewith=self.replacewith, ellipse=True, draw_scores=False, replaceimg=None, mosaicsize=15)
            # Write the frame to the output video file
            out.write(frame) if self.write_output else None
            frame_count += 1
            # Debugging information
            # self.logger.info(f"Frame {frame_count} written to video.")
            self.frame_emited.emit(frame)
            # Update progress bar
            progress = (frame_count / total_frames) * 100
            self.progress_updated.emit(int(progress))

            # Estimate remaining time
            elapsed_time = time.time() - start_time
            if frame_count > 0:
                frames_per_second = frame_count / elapsed_time
                remaining_frames = total_frames - frame_count
                remaining_time = remaining_frames / frames_per_second
                self.time_remaining_updated.emit(int(remaining_time))

        # Release resources
        cap.release()
        out.release() if self.write_output else None
        cv2.destroyAllWindows()
        self.anonymization_complete.emit()

    def pause(self):
        self._is_paused = True

    def resume(self):
        self._is_paused = False

    def stop(self):
        self._is_stopped = True

    def get_dimensions(self, format):

        if format == "480p":
            return 854, 480
        elif format == "720p":
            return 1280, 720
        elif format == "1080p":
            return 1920, 1080
        elif format == "360p":
            return 640, 360
        else:
            return 640, 480

    def yunet_mechanism(self, frame, mask_size, replacewith, ellipse=True, draw_scores=False, replaceimg=None, mosaicsize=15):
        # loop over the detections
        (h, w) = frame.shape[:2] # Get the height and width of the frame
        self.yunet.setInputSize((w, h))
        detections = self.yunet.infer(frame)
        # self.logger.info results
        # self.logger.info('{} faces detected.'.format(detections.shape[0]))
        # for idx, det in enumerate(detections):
        #     self.logger.info('{}: {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}'.format(
        #         idx, *det[:-1])
        #     )
        # if detections.shape[0] == 0:
        #     return frame
        frame = self.yunet.visualize(frame, detections, mask_size, replacewith, ellipse=True, draw_scores=False, replaceimg=None, mosaicsize=15)
        return frame

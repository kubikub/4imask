import sys
import cv2
import time
import subprocess
import tempfile
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QLabel, QProgressBar, QComboBox, QMessageBox
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, QThread, Signal
import numpy as np
import imutils
import os
from apps.yunet import YuNet
from apps.centerface import CenterFace
from typing import Tuple
import skimage.draw

class AnonymizationWorker(QThread):
    progress_updated = Signal(int)
    time_remaining_updated = Signal(int)
    anonymization_complete = Signal()
    frame_emited = Signal(np.ndarray)
    def __init__(self, video_path, output_format):
        super().__init__()
        self.video_path = video_path
        # self.net = net
        self.output_format = output_format

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

        fourcc = cv2.VideoWriter_fourcc(*'X264')
        out = cv2.VideoWriter('anonymized_video.mkv', fourcc, fps, (new_width, new_height))
        # out = cv2.VideoWriter('/home/frank-kubler/anonymized_video2.avi', fourcc, fps)
        if not out.isOpened():
            QMessageBox.critical(self, "Error", "Failed to open video writer.")
            cap.release()
            return
        centerface = CenterFace(in_shape=(new_width, new_height), backend='auto')
        # centerface.backend = 'onnxruntime-directml'
        start_time = time.time()
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = imutils.resize(frame, width=new_width)
            # frame = imutils.resize(frame, width=400)

            # grab the dimensions of the frame and then construct a blob
            # from it
            # (h, w) = frame.shape[:2]
            # blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
            #     (104.0, 177.0, 123.0))

            # pass the blob through the network and obtain the face detections
            # self.net.setInput(blob)
            # detections = self.net.forward()
                    # Inference
            #YUNET
            # self.net.setInputSize([w, h])
            # detections = self.net.infer(frame)
            
            detections, _ = centerface(frame, threshold=0.3)

            # loop over the detections
            # for i in range(0, detections.shape[2]):
            #     # extract the confidence (i.e., probability) associated with
            #     # the detection
            #     confidence = detections[0, 0, i, 2]

            #     # filter out weak detections by ensuring the confidence is
            #     # greater than the minimum confidence
            #     if confidence > 0.5:
            #         # compute the (x, y)-coordinates of the bounding box for
            #         # the object
            #         box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            #         (startX, startY, endX, endY) = box.astype("int")

            #         # extract the face ROI
            #         face = frame[startY:endY, startX:endX]
            #         face = self.anonymize_face_pixelate(face, blocks=20)

            #         # store the blurred face in the output image
            #         frame[startY:endY, startX:endX] = face
            # frame = self.visualize(frame, detections)
            self.anonymize_frame(detections, frame, mask_scale=1.3, replacewith='mosaic', ellipse=False, draw_scores=False, replaceimg=None, mosaicsize=15)
            out.write(frame)
            frame_count += 1
            # Debugging information
            # print(f"Frame {frame_count} written to video.")
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
        out.release()
        cv2.destroyAllWindows()
        self.anonymization_complete.emit()

    def get_dimensions(self, format):

        if format == "480p":
            return 854, 480
        elif format == "720p":
            return 1280, 720
        elif format == "1080p":
            return 1920, 1080
        elif format == "320p":
            return 320, 240
        else:
            return 640, 480

    def visualize(self, image, results, box_color=(0, 255, 0), text_color=(0, 0, 255), fps=None):
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
            cv2.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)

            conf = det[-1]
            cv2.putText(output, '{:.4f}'.format(conf), (bbox[0], bbox[1]+12), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color)

            landmarks = det[4:14].astype(np.int32).reshape((5, 2))
            for idx, landmark in enumerate(landmarks):
                cv2.circle(output, landmark, 2, landmark_color[idx], 2)

        return output

    def anonymize_frame(self, dets, frame, mask_scale,
                        replacewith, ellipse, draw_scores, replaceimg, mosaicsize
                        ):
        for i, det in enumerate(dets):
            boxes, score = det[:4], det[4]
            x1, y1, x2, y2 = boxes.astype(int)
            x1, y1, x2, y2 = self.scale_bb(x1, y1, x2, y2, mask_scale)
            # Clip bb coordinates to valid frame region
            y1, y2 = max(0, y1), min(frame.shape[0] - 1, y2)
            x1, x2 = max(0, x1), min(frame.shape[1] - 1, x2)
            self.draw_det(
                frame, score, i, x1, y1, x2, y2,
                replacewith=replacewith,
                ellipse=ellipse,
                draw_scores=draw_scores,
                replaceimg=replaceimg,
                mosaicsize=mosaicsize
            )

    def scale_bb(self, x1, y1, x2, y2, mask_scale=1.0):
        s = mask_scale - 1.0
        h, w = y2 - y1, x2 - x1
        y1 -= h * s
        y2 += h * s
        x1 -= w * s
        x2 += w * s
        return np.round([x1, y1, x2, y2]).astype(int)

    def draw_det(self, 
                 frame, score, det_idx, x1, y1, x2, y2,
                 replacewith: str = 'blur',
                 ellipse: bool = True,
                 draw_scores: bool = False,
                 ovcolor: Tuple[int] = (0, 0, 0),
                 replaceimg=None,
                 mosaicsize: int = 20):
        if replacewith == 'solid':
            cv2.rectangle(frame, (x1, y1), (x2, y2), ovcolor, -1)
        elif replacewith == 'blur':
            bf = 2  # blur factor (number of pixels in each dimension that the face will be reduced to)
            blurred_box =  cv2.blur(
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
        elif replacewith == 'img':
            target_size = (x2 - x1, y2 - y1)
            resized_replaceimg = cv2.resize(replaceimg, target_size)
            if replaceimg.shape[2] == 3:  # RGB
                frame[y1:y2, x1:x2] = resized_replaceimg
            elif replaceimg.shape[2] == 4:  # RGBA
                frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * (1 - resized_replaceimg[:, :, 3:] / 255) + resized_replaceimg[:, :, :3] * (resized_replaceimg[:, :, 3:] / 255)
        elif replacewith == 'mosaic':
            for y in range(y1, y2, mosaicsize):
                for x in range(x1, x2, mosaicsize):
                    pt1 = (x, y)
                    pt2 = (min(x2, x + mosaicsize - 1), min(y2, y + mosaicsize - 1))
                    color = (int(frame[y, x][0]), int(frame[y, x][1]), int(frame[y, x][2]))
                    cv2.rectangle(frame, pt1, pt2, color, -1)
        elif replacewith == 'none':
            pass
        if draw_scores:
            cv2.putText(
                frame, f'{score:.2f}', (x1 + 0, y1 - 20),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0)
            )



    def anonymize_face_pixelate(self, image, blocks=3):
        # divide the input image into NxN blocks
        (h, w) = image.shape[:2]
        xSteps = np.linspace(0, w, blocks + 1, dtype="int")
        ySteps = np.linspace(0, h, blocks + 1, dtype="int")

        # loop over the blocks in both the x and y direction
        for i in range(1, len(ySteps)):
            for j in range(1, len(xSteps)):
                # compute the starting and ending (x, y)-coordinates
                # for the current block
                startX = xSteps[j - 1]
                startY = ySteps[i - 1]
                endX = xSteps[j]
                endY = ySteps[i]

                # extract the ROI using NumPy array slicing, compute the
                # mean of the ROI, and then draw a rectangle with the
                # mean RGB values over the ROI in the original image
                roi = image[startY:endY, startX:endX]
                (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
                cv2.rectangle(image, (startX, startY), (endX, endY),
                    (B, G, R), -1)

        # return the pixelated blurred image
        return image


class VideoAnonymizer(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.ffmpeg_path = self.get_ffmpeg_path()
        print(self.ffmpeg_path)
        self.set_ffmpeg_env_path()
        print(self.is_ffmpeg_path_in_env())

        self.setWindowTitle("Video Anonymizer")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.select_button = QPushButton("Select Video")
        self.select_button.clicked.connect(self.select_video)
        self.layout.addWidget(self.select_button)

        self.format_label = QLabel("Select Output Format:")
        self.layout.addWidget(self.format_label)

        self.format_combo = QComboBox()
        self.format_combo.addItems(["480p", "720p", "1080p"])
        self.layout.addWidget(self.format_combo)

        self.start_button = QPushButton("Start Anonymization")
        self.start_button.clicked.connect(self.start_anonymization)
        self.layout.addWidget(self.start_button)

        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.progress_bar)

        self.time_label = QLabel("Estimated Time Remaining: --")
        self.layout.addWidget(self.time_label)

        self.notification_label = QLabel("")
        self.layout.addWidget(self.notification_label)

        self.original_label = QLabel("Original Video")
        self.layout.addWidget(self.original_label)

        self.anonymized_label = QLabel("Anonymized Video")
        self.layout.addWidget(self.anonymized_label)

        self.video_path = None
        self.cap = None
        self.timer = QTimer()
        # self.timer.timeout.connect(self.update_frame)
        self.pause_updates = False

    def select_video(self):
        options = QFileDialog.Options()
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)", options=options)
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.timer.start(30)
        else:
            QMessageBox.warning(self, "Warning", "No video selected.")

    def start_anonymization(self):
        # self.ffmpeg_path = self.get_ffmpeg_path()
        # self.set_ffmpeg_env_path()
        # print(self.is_ffmpeg_path_in_env())
        # self.net = self.load_model()
        if self.video_path:
            self.pause_updates = True
            output_format = self.format_combo.currentText()
            self.worker = AnonymizationWorker(self.video_path, output_format)
            self.worker.progress_updated.connect(self.update_progress)
            self.worker.time_remaining_updated.connect(self.update_time)
            self.worker.anonymization_complete.connect(self.anonymization_complete)
            self.worker.frame_emited.connect(self.update_frame)
            self.worker.start()
        else:
            QMessageBox.warning(self, "Warning", "Please select a video first.")

    def get_ffmpeg_path(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg", "bin")

    def set_ffmpeg_env_path(self):
        os.environ["PATH"] += os.pathsep + self.ffmpeg_path
        print(f"Updated PATH: {os.environ['PATH']}")

    def remove_ffmpeg_env_path(self):
        # ffmpeg_path = r"C:\Program Files\ffmpeg\bin"
        os.environ["PATH"] = os.pathsep.join(
            [p for p in os.environ["PATH"].split(os.pathsep) if p != self.ffmpeg_path]
        )
    def is_ffmpeg_path_in_env(self):
        return self.ffmpeg_path in os.environ["PATH"]

        
    # def load_model(self):
    #     # load our serialized face detector model from disk
    #     print("[INFO] loading face detector model...")
    #     # prototxtPath = os.path.join('ressources/models', "deploy.prototxt")
    #     # weightsPath = os.path.join('ressources/models',
    #     #     "res10_300x300_ssd_iter_140000.caffemodel")
    #     # net = cv2.dnn.readNet(prototxtPath, weightsPath)
    #     # print("[INFO] model loaded.")
    #     # net = YuNet("ressources/models/face_detection_yunet_2023mar.onnx")
    #     net = CenterFace()
    #     return net

    def update_progress(self, progress):
        self.progress_bar.setValue(progress)

    # def update_time(self, remaining_time):
        # self.time_label.setText(f"Estimated Time Remaining: {remaining_time} seconds")

    def update_time(self, remaining_time):
        hours, remainder = divmod(remaining_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"Estimated Time Remaining: {int(hours)}h {int(minutes)}m {int(seconds)}s"
        self.time_label.setText(time_str)

    def anonymization_complete(self):
        self.notification_label.setText("Anonymization complete.")

    def update_frame(self, frame):
        if self.pause_updates:
            if not hasattr(self, 'paused_frame_displayed') or not self.paused_frame_displayed:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = convert_to_Qt_format.scaled(720, 480)
                self.original_label.setPixmap(QPixmap.fromImage(p))
                self.paused_frame_displayed = True
            return
        
        # Reset the flag when updates are not paused
        self.paused_frame_displayed = False
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(720, 480)
        self.original_label.setPixmap(QPixmap.fromImage(p))

    def closeEvent(self, event):
        self.remove_ffmpeg_env_path()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoAnonymizer()
    window.show()
    sys.exit(app.exec())
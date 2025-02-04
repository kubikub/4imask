import sys
import cv2
import time
import subprocess
import tempfile
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog,
                                QHBoxLayout, QVBoxLayout, QWidget, QLabel, QProgressBar, QComboBox, QMessageBox, QCheckBox)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, QThread, Signal
import numpy as np
import imutils
import os
from apps.yunet import YuNet
from apps.centerface import CenterFace
from typing import Tuple
import skimage.draw
import platform
from apps.yolo import YOLOModel
from apps.yolo import blur_faces
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
        self._is_paused = False
        self._is_stopped = False
        self.replacewith = 'mosaic'  # Default value

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
        print(self.output_format)
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        # output_name = os.path.basename(self.video_path).split(".")[0] + "_anonymized_" + self.output_format.lower() + ".mkv"
        output_name = (self.video_path).split(".")[0] + "_anonymized_" + self.output_format.lower() + ".mkv"

        print(output_name)
        out = cv2.VideoWriter(output_name, fourcc, fps, (new_width, new_height))
        # out = cv2.VideoWriter('/home/frank-kubler/anonymized_video2.avi', fourcc, fps)
        if not out.isOpened():
            QMessageBox.critical(self, "Error", "Failed to open video writer.")
            cap.release()
            return
        centerface = CenterFace(in_shape=(new_width, new_height), backend='auto')  # auto
        yolo_model = YOLOModel()

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
            
            faces = yolo_model.detect_faces(frame, 0.25, 0.45)
            if faces is not None:
                frame = blur_faces(faces, frame, 20)
            # detections, _ = centerface(frame, threshold=0.4)
            # self.anonymize_frame(detections, frame, mask_scale=1.3, replacewith=self.replacewith, ellipse=False, draw_scores=False, replaceimg=None, mosaicsize=15)
            
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
        self.ffmpeg_env = FFmpegPathManager(self.ffmpeg_path)
        self.ffmpeg_env.set_ffmpeg_env_path()
        print(self.ffmpeg_env.is_ffmpeg_path_in_env())

        self.setWindowTitle("Video Anonymizer")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.select_button = QPushButton("Select Video")
        self.select_button.clicked.connect(self.select_video)
        self.video_label = QLabel("No video selected.")
        self.layout.addWidget(self.select_button)
        self.layout.addWidget(self.video_label)

        self.format_label = QLabel("Select Output Format:")
        self.layout.addWidget(self.format_label)

        self.format_combo = QComboBox()
        self.format_combo.addItems(["480p", "360p", "720p", "1080p"])
        self.layout.addWidget(self.format_combo)

        self.anonymization_options_layout = QHBoxLayout()
        self.mosaic_checkbox = QCheckBox("Mosaic")
        self.mosaic_checkbox.setChecked(True)
        self.blur_checkbox = QCheckBox("Blur")
        self.mask_checkbox = QCheckBox("Mask")
        self.mosaic_checkbox.toggled.connect(self.update_checkboxes)
        self.blur_checkbox.toggled.connect(self.update_checkboxes)
        self.mask_checkbox.toggled.connect(self.update_checkboxes)
        self.anonymization_options_layout.addWidget(self.mosaic_checkbox)
        self.anonymization_options_layout.addWidget(self.blur_checkbox)
        self.anonymization_options_layout.addWidget(self.mask_checkbox)
        self.layout.addLayout(self.anonymization_options_layout)

        self.buttons_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Anonymization")
        self.start_button.clicked.connect(self.start_anonymization)
        self.buttons_layout.addWidget(self.start_button)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_anonymization)
        self.buttons_layout.addWidget(self.pause_button)

        self.resume_button = QPushButton("Resume")
        self.resume_button.clicked.connect(self.resume_anonymization)
        self.buttons_layout.addWidget(self.resume_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_anonymization)
        self.buttons_layout.addWidget(self.stop_button)

        self.play_button = QPushButton("Play Preview")
        self.play_button.clicked.connect(self.play_anonymization)
        self.buttons_layout.addWidget(self.play_button)

        self.layout.addLayout(self.buttons_layout)

        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.progress_bar)

        self.time_label = QLabel("Estimated Time Remaining: --")
        self.layout.addWidget(self.time_label)

        # self.layout.addWidget(self.play_button)
        self.notification_label = QLabel("")
        self.layout.addWidget(self.notification_label)

        self.original_label = QLabel("Original Video")
        self.layout.addWidget(self.original_label)

        # self.anonymized_label = QLabel("Anonymized Video")
        # self.layout.addWidget(self.anonymized_label)

        self.video_path = None
        self.cap = None
        self.timer = QTimer()
        # self.timer.timeout.connect(self.update_frame)
        self.pause_updates = False

    def select_video(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            QMessageBox.warning(self, "Warning", "Anonymization is in progress. Please stop it first.")
            return
        options = QFileDialog.Options()
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)", options=options)
        self.video_label.setText(os.path.basename(self.video_path))
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
            replacewith = self.get_replacewith_option()
            self.worker = AnonymizationWorker(self.video_path, output_format)
            self.worker.replacewith = replacewith
            self.worker.progress_updated.connect(self.update_progress)
            self.worker.time_remaining_updated.connect(self.update_time)
            self.worker.anonymization_complete.connect(self.anonymization_complete)
            self.worker.frame_emited.connect(self.update_frame)
            self.worker.start()
            self.select_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.resume_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.blur_checkbox.setEnabled(False)
            self.mask_checkbox.setEnabled(False)
            self.mosaic_checkbox.setEnabled(False)
            self.play_button.setEnabled(True)

        else:
            QMessageBox.warning(self, "Warning", "Please select a video first.")

    def update_checkboxes(self, checked):
        if checked:
            sender = self.sender()
            if sender == self.mosaic_checkbox:
                self.blur_checkbox.setChecked(False)
                self.mask_checkbox.setChecked(False)
            elif sender == self.blur_checkbox:
                self.mosaic_checkbox.setChecked(False)
                self.mask_checkbox.setChecked(False)
            elif sender == self.mask_checkbox:
                self.mosaic_checkbox.setChecked(False)
                self.blur_checkbox.setChecked(False)

    def get_replacewith_option(self):
        if self.mosaic_checkbox.isChecked():
            return 'mosaic'
        elif self.blur_checkbox.isChecked():
            return 'blur'
        elif self.mask_checkbox.isChecked():
            return 'solid'
        else:
            return 'none'

    def pause_anonymization(self):
        if hasattr(self, 'worker'):
            self.worker.pause()
            self.pause_button.setEnabled(False)
            self.resume_button.setEnabled(True)

    def resume_anonymization(self):
        if hasattr(self, 'worker'):
            self.worker.resume()
            self.pause_button.setEnabled(True)
            self.resume_button.setEnabled(False)

    def stop_anonymization(self):
        if hasattr(self, 'worker'):
            self.worker.stop()
            self.select_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.resume_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.blur_checkbox.setEnabled(True)
            self.mask_checkbox.setEnabled(True)
            self.mosaic_checkbox.setEnabled(True)
            self.play_button.setEnabled(False)

    def get_ffmpeg_path(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg", "bin")

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
        self.select_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.play_button.setEnabled(False)

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

    def play_anonymization(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)
        else:
            QMessageBox.warning(self, "Warning", "Anonymization is not in progress. Please start the anonymization first.")

    def closeEvent(self, event):
        self.ffmpeg_env.remove_ffmpeg_env_path()
        event.accept()

class FFmpegPathManager:
    def __init__(self, ffmpeg_path):
        self.ffmpeg_path = ffmpeg_path

    def set_ffmpeg_env_path(self):
        if platform.system() == "Windows":
            os.environ["PATH"] += os.pathsep + self.ffmpeg_path
        else:
            os.environ["PATH"] += os.pathsep + self.ffmpeg_path
        print(f"Updated PATH: {os.environ['PATH']}")

    def remove_ffmpeg_env_path(self):
        os.environ["PATH"] = os.pathsep.join(
            [p for p in os.environ["PATH"].split(os.pathsep) if p != self.ffmpeg_path]
        )

    def is_ffmpeg_path_in_env(self):
        return self.ffmpeg_path in os.environ["PATH"]


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoAnonymizer()
    window.show()
    sys.exit(app.exec())
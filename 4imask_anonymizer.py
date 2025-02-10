import sys
import cv2
import platform
import qdarktheme
import os
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog,
                                QDoubleSpinBox, QSpacerItem, QSizePolicy, 
                                QHBoxLayout, QVBoxLayout, QWidget, QLabel,
                                QProgressBar, QComboBox, QMessageBox, QCheckBox)
from PySide6.QtGui import QImage, QPixmap, QIcon
from PySide6.QtCore import QTimer
from apps.worker import AnonymizationWorker
from apps.utils import resource_path
import logging
from datetime import datetime


try:
    import pyi_splash
    pyi_splash.update_text("Loading...")
    pyi_splash.close()
except ImportError:
    print("pyi_splash not found, continuing without splash screen.")

class VideoAnonymizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.version = "1.0.0"
        self.setWindowIcon(QIcon(resource_path("res/icons/4imask.png")))
        self.setWindowTitle("4iMask Anonymizer")
        self.logger = logging.getLogger(__name__).getChild(self.__class__.__name__)
        variable_name = "USERDOMAIN"
        value = os.getenv(variable_name)
        authorized_domains = open(resource_path("res/.dummy"), "r").read().splitlines()
        if value not in authorized_domains:
            QMessageBox.warning(self, "Warning", f"You are not authorized to use this application.")
            sys.exit()
        
        # self.ffmpeg_path = self.get_ffmpeg_path()
        # self.ffmpeg_env = FFmpegPathManager(self.ffmpeg_path)
        # self.ffmpeg_env.set_ffmpeg_env_path()
        # self.logger.info("FFmpeg path in environment: ", self.ffmpeg_env.is_ffmpeg_path_in_env())

        self.setWindowTitle("4iMask Anonymizer")
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
        self.mosaic_checkbox.setEnabled(True)
        self.blur_checkbox = QCheckBox("Blur")
        self.blur_checkbox.setChecked(True)
        self.mask_checkbox = QCheckBox("Mask")
        self.mask_checkbox.setEnabled(True)
        self.mosaic_checkbox.toggled.connect(self.update_checkboxes)
        self.blur_checkbox.toggled.connect(self.update_checkboxes)
        self.mask_checkbox.toggled.connect(self.update_checkboxes)
        self.anonymization_options_layout.addWidget(self.mosaic_checkbox)
        self.anonymization_options_layout.addWidget(self.blur_checkbox)
        self.anonymization_options_layout.addWidget(self.mask_checkbox)
        self.spin_label = QLabel("Mask Size:")
        self.anonymization_options_layout.addWidget(self.spin_label)
        self.spinbox = QDoubleSpinBox()
        self.spinbox.setMinimum(1.0)
        self.spinbox.setMaximum(5.0)
        self.spinbox.setSingleStep(0.1)
        self.spinbox.setValue(1.3)
        self.spinbox.valueChanged.connect(self.update_mask_size)

        self.anonymization_options_layout.addWidget(self.spinbox)
        self.layout.addLayout(self.anonymization_options_layout)
        
        self.detection_models_layout = QHBoxLayout()
        self.yolo_checkbox = QCheckBox("Yolov11n Face")
        self.yolo_checkbox.setChecked(True)
        self.centerface_checkbox = QCheckBox("Center Face")
        self.yunet_checkbox = QCheckBox("Yunet")
        self.yolo_checkbox.toggled.connect(self.update_model_checkboxes)
        self.centerface_checkbox.toggled.connect(self.update_model_checkboxes)
        self.yunet_checkbox.toggled.connect(self.update_model_checkboxes)
        self.detection_models_layout.addWidget(self.yolo_checkbox)
        self.detection_models_layout.addWidget(self.centerface_checkbox)
        self.detection_models_layout.addWidget(self.yunet_checkbox)
        self.layout.addLayout(self.detection_models_layout)
        
        self.buttons_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Anonymization")
        self.start_button.clicked.connect(self.start_anonymization)
        self.buttons_layout.addWidget(self.start_button)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_anonymization)
        self.pause_button.setEnabled(False)
        self.buttons_layout.addWidget(self.pause_button)

        self.resume_button = QPushButton("Resume")
        self.resume_button.clicked.connect(self.resume_anonymization)
        self.resume_button.setEnabled(False)
        self.buttons_layout.addWidget(self.resume_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_anonymization)
        self.stop_button.setEnabled(False)
        self.buttons_layout.addWidget(self.stop_button)

        self.play_button = QPushButton("Play Preview")
        self.play_button.clicked.connect(self.toggle_play_pause)
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
        # Ajouter un espaceur flexible pour pousser l'étiquette de version vers le bas
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.layout.addItem(spacer)
        self.version_label = QLabel(f"2025 4iMask Anonymizer Version: {self.version}")
        self.layout.addWidget(self.version_label)

        # self.anonymized_label = QLabel("Anonymized Video")
        # self.layout.addWidget(self.anonymized_label)

        self.video_path = None
        self.cap = None
        self.timer = QTimer()
        # self.timer.timeout.connect(self.update_frame)
        self.pause_updates = False
        self.is_playing = False

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
        # self.logger.info(self.is_ffmpeg_path_in_env())
        # self.net = self.load_model()
        if self.video_path:
            self.pause_updates = True
            output_format = self.format_combo.currentText()
            replacewith = self.get_replacewith_option()
            model = self.get_model_option()
            if hasattr(self, 'worker') and self.worker.isRunning():
                self.worker.stop()
                self.worker.wait()
            self.worker = AnonymizationWorker(self.video_path, output_format, write_output=True)
            self.logger.info(f'Model: {model}')
            self.logger.info(f'Replacewith: {replacewith}')
            self.worker.model = model
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
            self.play_button.setEnabled(False)  # Disable play button during anonymization
            self.yolo_checkbox.setEnabled(False)
            self.centerface_checkbox.setEnabled(False)
            self.yunet_checkbox.setEnabled(False)
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
        self.get_replacewith_option()

    def update_model_checkboxes(self, checked):
        if checked:
            sender = self.sender()
            if sender == self.yolo_checkbox:
                self.centerface_checkbox.setChecked(False)
                self.yunet_checkbox.setChecked(False)
            elif sender == self.centerface_checkbox:
                self.yolo_checkbox.setChecked(False)
                self.yunet_checkbox.setChecked(False)
            elif sender == self.yunet_checkbox:
                self.yolo_checkbox.setChecked(False)
                self.centerface_checkbox.setChecked(False)
        self.get_model_option()
    
    def get_model_option(self):
        if self.yolo_checkbox.isChecked():
            if hasattr(self, 'worker'):
                self.worker.model = 'yolo'
            return 'yolo'
        elif self.centerface_checkbox.isChecked():
            if hasattr(self, 'worker'):
                self.worker.model = 'centerface'
            return 'centerface'
        elif self.yunet_checkbox.isChecked():
            if hasattr(self, 'worker'):
                self.worker.model = 'yunet'
            return 'yunet'
        else:
            return 'none'

    def update_mask_size(self, value):
        self.logger.info(value)
        if hasattr(self, 'worker'):
            self.worker.mask_size = value
        else:
            return value

    def get_replacewith_option(self):
        if self.mosaic_checkbox.isChecked():
            if hasattr(self, 'worker'):
                self.worker.replacewith = 'mosaic'
            return 'mosaic'
        elif self.blur_checkbox.isChecked():
            if hasattr(self, 'worker'):
                self.worker.replacewith = 'blur'
            return 'blur'
        elif self.mask_checkbox.isChecked():
            if hasattr(self, 'worker'):
                self.worker.replacewith = 'solid'
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
            self.yolo_checkbox.setEnabled(True)
            self.centerface_checkbox.setEnabled(True)
            self.yunet_checkbox.setEnabled(True)
            self.play_button.setEnabled(True)  # Enable play button after stopping
            self.play_button.setText("Play Preview")
            
            self.is_playing = False

    # def get_ffmpeg_path(self):
    #     return os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg", "bin")

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
        self.play_button.setEnabled(True)
        self.yolo_checkbox.setEnabled(True)
        self.centerface_checkbox.setEnabled(True)
        self.yunet_checkbox.setEnabled(True)
        self.play_button.setText("Play Preview")
        self.is_playing = False

    def update_frame(self, frame=None):
        if not self.pause_updates and frame is not None:
        #     if not hasattr(self, 'paused_frame_displayed') or not self.paused_frame_displayed:
        #         rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #         h, w, ch = rgb_image.shape
        #         bytes_per_line = ch * w
        #         convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        #         p = convert_to_Qt_format.scaled(720, 480)
        #         self.original_label.setPixmap(QPixmap.fromImage(p))
        #         self.paused_frame_displayed = True
        #     return

        # # Reset the flag when updates are not paused
        # self.paused_frame_displayed = False
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(720, 480)
            self.original_label.setPixmap(QPixmap.fromImage(p))

    def toggle_play_pause(self):
        if self.is_playing:
            self.stop_preview()
            self.start_button.setEnabled(True)
        else:
            self.play_preview()
            self.start_button.setEnabled(False)

    def play_preview(self):
        if self.video_path:
            self.pause_updates = False
            output_format = self.format_combo.currentText()
            replacewith = self.get_replacewith_option()
            model = self.get_model_option()
            if hasattr(self, 'worker') and self.worker.isRunning():
                self.logger.info(self.worker.isRunning())

                self.worker.stop()
                self.worker.quit()
                self.worker.wait()
                self.logger.info("Worker stopped.")
            self.worker = AnonymizationWorker(self.video_path, output_format, write_output=False)
            self.worker.model = model
            self.worker.replacewith = replacewith
            self.logger.info(f'Model  : {model}')
            self.logger.info(f'Replacewith 3: {replacewith}')
            self.worker.frame_emited.connect(self.update_frame)
            self.worker.start()
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)
            self.play_button.setText("Stop Preview")
            self.is_playing = True
            self.stop_button.setEnabled(False)
            self.select_button.setEnabled(False)
            self.format_combo.setEnabled(False)
            self.resume_button.setEnabled(False)
            self.pause_button.setEnabled(False)
            self.yolo_checkbox.setEnabled(False)
            self.centerface_checkbox.setEnabled(False)
            self.yunet_checkbox.setEnabled(False)
        else:
            QMessageBox.warning(self, "Warning", "Please select a video first.")

    def stop_preview(self):
        self.timer.stop()
        self.worker.stop()
        self.worker.quit()
        self.worker.wait()
        
        self.pause_updates = True
        self.play_button.setText("Play Preview")
        self.is_playing = False
        self.select_button.setEnabled(True)
        self.format_combo.setEnabled(True)
        self.yolo_checkbox.setEnabled(True)
        self.centerface_checkbox.setEnabled(True)
        self.yunet_checkbox.setEnabled(True)

        # self.resume_button.setEnabled(True)
        # self.pause_button.setEnabled(True)

    def closeEvent(self, event):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        # self.ffmpeg_env.remove_ffmpeg_env_path()
        event.accept()

# class FFmpegPathManager:
#     def __init__(self, ffmpeg_path):
#         self.ffmpeg_path = ffmpeg_path

#     def set_ffmpeg_env_path(self):
#         if platform.system() == "Windows":
#             os.environ["PATH"] += os.pathsep + self.ffmpeg_path
#         else:
#             os.environ["PATH"] += os.pathsep + self.ffmpeg_path
#         # self.logger.info(f"Updated PATH: {os.environ['PATH']}")

#     def remove_ffmpeg_env_path(self):
#         os.environ["PATH"] = os.pathsep.join(
#             [p for p in os.environ["PATH"].split(os.pathsep) if p != self.ffmpeg_path]
#         )

#     def is_ffmpeg_path_in_env(self):
#         return self.ffmpeg_path in os.environ["PATH"]

def mainloop():
    qdarktheme.enable_hi_dpi()
    logs_settings()
    app = QApplication(sys.argv)
    path = resource_path("res/icons/4imask.png")
    app.setWindowIcon(QIcon(path))
    app.setStyle('Fusion')
    qdarktheme.setup_theme("light")
    window = VideoAnonymizer()
    window.show()
    sys.exit(app.exec())

def logs_settings():
    try:
        os.mkdir('logs')
    except FileExistsError:
        pass
    log_dir = 'logs/'
    # List all log files in the directory
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]

    # Sort the log files by creation time
    log_files.sort(key=lambda f: os.path.getctime(os.path.join(log_dir, f)))

    # Keep only the last 10 files
    if len(log_files) > 10:
        for log_file in log_files[:-10]:
            os.remove(os.path.join(log_dir, log_file))

    # Configure the root logger to log messages to a file and the console
    # file_handler = logging.FileHandler('logs/' + 'app.log')
    log_file_path = os.path.join(log_dir, datetime.now().strftime('%Y-%m-%d %H-%M-%S.log'))
    file_handler = logging.FileHandler(log_file_path)
    console_handler = logging.StreamHandler(sys.stdout)

    logging.basicConfig(level=logging.INFO,
                        format='Line: %(lineno)d - %(message)s - %(levelname)s - %(name)s',
                        handlers=[file_handler, console_handler])
    # Suppress all Matplotlib loggers
    for logger_name in logging.Logger.manager.loggerDict:
        if logger_name.startswith('matplotlib'):
            logging.getLogger(logger_name).setLevel(logging.WARNING)
    # Function to log uncaught exceptions

    def log_uncaught_exceptions(exctype, value, tb):
        logging.error("Uncaught exception", exc_info=(exctype, value, tb))
        file_handler.flush()
    # Set the exception hook
    sys.excepthook = log_uncaught_exceptions
    # Flush and close the log file
    file_handler.flush()
    file_handler.close()


if __name__ == "__main__":
    mainloop()
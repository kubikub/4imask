# 4iMask Anonymizer
## Description
4iMask Anonymizer is an application for anonymizing videos using various techniques such as blurring, masking, and mosaicking.

## Features
- Select video to anonymize
- Choose output format (360p, 480p, 720p, 1080p)
- Anonymization options: Mosaic, Blur, Mask
- Pause, resume, and stop anonymization
- Live view of anonymization
- Change detection model (YOLOv11, CenterFace, YUNET)

## Prerequisites
- Python 3.x
- PySide6
- OpenCV
- imutils
- skimage
- onnx

For Windows:
- onnxruntime-directml

For Linux:
- onnxruntime-openvino

With NVIDIA GPU:
- onnxruntime-gpu

## Installation
Clone the repository and install the dependencies:
```sh
git clone https://github.com/your-username/4imask.git
cd 4imask
pip install -r requirements.txt
```

## Usage
To launch the application, run:
```sh
python 4imask_anonymizer.py
```

## Compilation
Ensure all dependencies are installed using `pip install -r requirements.txt`. Then, you can run the application with the command above.

## Contribute
Contributions are welcome. Please open an issue or submit a pull request.

## License
This project is licensed under AGPL-3.0. See the [LICENSE](./LICENSE) file for details.

The libraries used in this project are under the following licenses:
- OpenCV: Apache 2.0
- PySide6: LGPL
- imutils: MIT
- skimage: BSD
- onnx: MIT
- onnxruntime-directml: MIT
- onnxruntime-openvino: MIT
- onnxruntime-gpu: MIT
- Ultralytics: AGPL-3.0

## Contact
For any questions, please contact via github issues.

## Sources
- [OpenCV Zoo - Face Detection Yunet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- [ORB-HD - Deface](https://github.com/ORB-HD/deface)
- [Akanametov - YOLO Face](https://github.com/akanametov/yolo-face)
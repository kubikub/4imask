import os
import sys
# import skimage.draw
from skimage.draw import ellipse as skimage_ellipse
import cv2
import numpy as np
# def resource_path(relative_path):
#     """ Get absolute path to resource, works for dev and for PyInstaller """
#     base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
#     return os.path.join(base_path, relative_path)

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def draw_detections(frame, x1, y1, x2, y2,
                    replacewith : str = 'blur', mask_size : float = 1.3,
                    ellipse : bool = True, 
                    mosaicsize : int = 15):
            # Scale bb coordinates to mask size
            x1, y1, x2, y2 = scale_bb(x1, y1, x2, y2, mask_size)
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
                    ey, ex = skimage_ellipse((y2 - y1) // 2, (x2 - x1) // 2, (y2 - y1) // 2, (x2 - x1) // 2)
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

def scale_bb(x0, y0, x1, y1, mask_scale=1.3):
    s = mask_scale - 1.0
    h, w = y1 - y0, x1 - x0
    y0 -= h * s
    y1 += h * s
    x0 -= w * s
    x1 += w * s
    return np.round([x0, y0, x1, y1]).astype(int)


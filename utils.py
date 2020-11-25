import cv2 as cv
import numpy as np

def pil_to_opencv(image):
    return cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)

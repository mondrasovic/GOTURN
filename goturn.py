import cv2 as cv

from got10k.trackers import Tracker

from utils import pil_to_opencv


class TrackerGOTURN(Tracker):
    def __init__(self):
        super().__init__(name='GOTURN', is_deterministic=True)
        
        self._tracker = cv.TrackerGOTURN_create()
    
    def init(self, image, box):
        self._tracker.init(pil_to_opencv(image), tuple(box))
    
    def update(self, image):
        print(f'{image}')
        print(f'shape: {pil_to_opencv(image).shape}')
        _, box = self._tracker.update(pil_to_opencv(image))
        return box

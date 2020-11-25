import cv2 as cv

from got10k.trackers import Tracker


class TrackerGOTURN(Tracker):
    def __init__(self):
        super().__init__(name='GOTURN', is_deterministic=True)
    
    def init(self, image, box):
        self.box = box
    
    def update(self, image):
        return self.box
import cv2 as cv

def main():
    tracker = cv.TrackerGOTURN_create()
    capture = cv.VideoCapture(0)
    
    _, frame = capture.read()
    bbox = cv.selectROI(frame, False)
    tracker.init(frame, bbox)
    
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        timer = cv.getTickCount()
        ok, bbox = tracker.update(frame)
        fps = cv.getTickFrequency() / (cv.getTickCount() - timer)
        
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            cv.putText(
                frame, 'Tracking failure detected', (100, 80), cv.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 255), 2)
        
        cv.putText(
            frame, 'GOTURN Tracker', (100, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv.putText(
            frame, 'FPS : ' + str(int(fps)), (100, 50), cv.FONT_HERSHEY_SIMPLEX, 0.75,
            (50, 170, 50), 2)
        
        cv.imshow('Tracking', frame)
        key = cv.waitKey(1) & 0xff
        if key == ord('q'):
            break

    return 0


if __name__ == '__main__':
    import sys
    
    sys.exit(main())

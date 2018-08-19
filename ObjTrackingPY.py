import numpy as np
import cv2

def BGwithVersion(v):
    return {
    '2' : cv2.BackgroundSubtractorMOG2,
    '3' : cv2.createBackgroundSubtractorMOG2()
    }[v]

def main():
    Cap = cv2.VideoCapture(video path)
    ret ,_ = Cap.read()
    if not ret:
        print('Could not read or open the file !!')
        return
    cv2.ocl.setUseOpenCL(True)
    BG = BGwithVersion(cv2.__version__.split('.')[0])
    minArea = 1000
    while (Cap.isOpened):
        ret, frame = Cap.read()
        if not ret:
            break
        mask = BG.apply(frame)
        (_, Contours ,_) = cv2.findContours(mask.copy(),
                                            cv2.RETR_LIST,
                                            cv2.CHAIN_APPROX_SIMPLE)
        for area in Contours:
            if cv2.contourArea(area) < minArea:
                continue
            (x, y, w, h) = cv2.boundingRect(area)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (127, 0, 200), 3)
        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    Cap.release()
    cv2.destroyAllWindows()
    return

if __name__ == "__main__":
    main()

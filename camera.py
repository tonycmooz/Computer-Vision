import cv2
import pandas
from datetime import datetime
import time

video = cv2.VideoCapture(1)

first_frame = None
status_list = [None, None]
# times = []
# df = pandas.DataFrame(columns=["Start", "End"])

while True:
    check, frame = video.read()
    status = 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame, gray)
    thresh_delta = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]  # play with
    thresh_delta = cv2.dilate(thresh_delta, None, iterations=2)
    (_, cnts, _) = cv2.findContours(thresh_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Rectangle
    for contour in cnts:
        if cv2.contourArea(contour) < 10000:  # play with
            continue
        status = 1
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
    status_list.append(status)
    status_list = status_list[-2:]

    # # To keep track of time deltas
    # if status_list[-1] == 1 and status_list[-2] == 0:
    #     times.append(datetime.now())
    # if status_list[-1] == 0 and status_list[-2] == 1:
    #     times.append(datetime.now())

    # Present frames
    cv2.imshow('frame', frame)
    # cv2.imshow('Showing', gray)
    # cv2.imshow('delta', delta_frame)
    # cv2.imshow('threshold', thresh_delta)

    # Quit program
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# print(a)
video.release()
cv2.destroyAllWindows()
print('Done Capturing Video')

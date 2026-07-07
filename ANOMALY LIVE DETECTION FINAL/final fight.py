import cv2
import numpy as np
from ultralytics import YOLO
from playsound import playsound
import threading

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("videos/crowd_video.mp4")

prev = {}
alarm_on = False

def play_alarm():
    threading.Thread(target=playsound, args=("videos/voice.mp3",), daemon=True).start()

def center(b):
    x1,y1,x2,y2 = b
    return np.array([(x1+x2)/2, (y1+y2)/2])

def dist(a,b):
    return np.linalg.norm(a-b)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (500,500))

    results = model.track(frame, persist=True, classes=[0], conf=0.25)

    if results[0].boxes.id is None:
        cv2.imshow("Fight Detect", frame)
        if cv2.waitKey(1)==27:
            break
        continue

    boxes = results[0].boxes.xyxy.cpu().numpy()
    ids = results[0].boxes.id.cpu().numpy()

    curr = {}
    fighting = set()

    for box, pid in zip(boxes, ids):
        curr[pid] = (center(box), box)

    id_list = list(curr.keys())

    # 🔴 pair interaction detection
    for i in range(len(id_list)):
        for j in range(i+1, len(id_list)):
            id1, id2 = id_list[i], id_list[j]
            p1, b1 = curr[id1]
            p2, b2 = curr[id2]

            d = dist(p1,p2)
            s1 = dist(p1, prev.get(id1,p1))
            s2 = dist(p2, prev.get(id2,p2))

            if d < 80 and (s1+s2) > 6:
                fighting.add(id1)
                fighting.add(id2)

                if not alarm_on:
                    play_alarm()
                    alarm_on = True
            else:
                alarm_on = False

    # 🎯 draw boxes
    for pid in curr:
        box = curr[pid][1]
        x1,y1,x2,y2 = map(int, box)

        if pid in fighting:
            color = (0,0,255)
            text = f"FIGHT ID {int(pid)}"
        else:
            color = (0,255,0)
            text = f"ID {int(pid)}"

        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        cv2.putText(frame,text,(x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    prev = {pid: curr[pid][0] for pid in curr}

    cv2.imshow("Fight Detect", frame)
    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()

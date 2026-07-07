import cv2
import numpy as np
from ultralytics import YOLO
from playsound import playsound
import threading

# 🔹 YOLOv8 medium model
model = YOLO("yolov8m.pt")

# 🔹 Video input
cap = cv2.VideoCapture("videos/crowd_video1.mp4")

# 🔹 Previous positions & alarm
prev_positions = {}
alarm_on = False

# 🔹 Alarm thread
def play_alarm():
    threading.Thread(target=playsound, args=("videos/voice.mp3",), daemon=True).start()

# 🔹 Helper functions
def center(box):
    x1,y1,x2,y2 = box
    return np.array([(x1+x2)/2, (y1+y2)/2])

def dist(a,b):
    return np.linalg.norm(a-b)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (720,720))

    results = model.track(frame, persist=True, classes=[0], conf=0.4)

    if results[0].boxes.id is None:
        cv2.imshow("Abnormal Activity Detection", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    boxes = results[0].boxes.xyxy.cpu().numpy()
    ids = results[0].boxes.id.cpu().numpy()

    curr = {}
    abnormal_ids = set()

    for box, pid in zip(boxes, ids):
        curr[pid] = (center(box), box)

    id_list = list(curr.keys())

    # 🔹 compute distance + speed-based abnormality score
    for i in range(len(id_list)):
        for j in range(i+1, len(id_list)):
            id1, id2 = id_list[i], id_list[j]
            p1, _ = curr[id1]
            p2, _ = curr[id2]

            # distance between people
            d = dist(p1,p2)

            # speed from previous frame
            s1 = dist(p1, prev_positions.get(id1,p1))
            s2 = dist(p2, prev_positions.get(id2,p2))

            # 🔹 combined abnormality score
            # fight score: closer + moving fast
            fight_score = (max(0, 50 - d)) + (s1 + s2)
            
            # running score: speed weighted
            run_score1 = s1
            run_score2 = s2

            # jump score: vertical displacement
            jump_score1 = abs(p1[1] - prev_positions.get(id1,p1)[1])
            jump_score2 = abs(p2[1] - prev_positions.get(id2,p2)[1])

            # thresholds for alert
            if fight_score > 30 or run_score1 > 10 or run_score2 > 10 or jump_score1 > 10 or jump_score2 > 10:
                abnormal_ids.add(id1)
                abnormal_ids.add(id2)

    # 🔔 alarm
    if abnormal_ids and not alarm_on:
        play_alarm()
        alarm_on = True
    elif not abnormal_ids:
        alarm_on = False

    # 🔹 draw bounding boxes
    for pid in curr:
        box = curr[pid][1]
        x1,y1,x2,y2 = map(int, box)

        if pid in abnormal_ids:
            color = (0,0,255)
            text = f"Thief ID {int(pid)}"
        else:
            color = (0,255,0)
            text = f"ID {int(pid)}"

        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        cv2.putText(frame,text,(x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    prev_positions = {pid: curr[pid][0] for pid in curr}

    cv2.imshow("Abnormal Activity Detection", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

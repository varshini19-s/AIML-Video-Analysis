import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from ultralytics import YOLO
from playsound import playsound
import threading
import mediapipe as mp
import time

# -----------------------------
# Load YOLOv8 model
# -----------------------------
model = YOLO("yolov8s.pt")

# -----------------------------
# Pose for punch detection
# -----------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

prev_wrist = None
prev_centroids = []

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

ALERT_THRESHOLD = 5
CLUSTER_DISTANCE = 75
CLUSTER_SIZE_THRESHOLD = 3

ret, frame = cap.read()
h, w = frame.shape[:2]
heatmap = np.zeros((h, w), dtype=np.float32)

# -----------------------------
# Attack control parameters
# -----------------------------
ATTACK_CONFIRM_FRAMES = 6
attack_counter = 0

ALARM_COOLDOWN = 3
last_alarm_time = 0

MIN_INTERACTION_DISTANCE = 150
PUNCH_SPEED_THRESHOLD = 120
PUSH_DISTANCE_THRESHOLD = 120
PUSH_SPEED_THRESHOLD = 45

# -----------------------------
# Sound
# -----------------------------
def play_attack_sound():
    playsound("voice.mp3", block=False)

# -----------------------------
# Main Loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    results = model(frame, verbose=False)

    people_centroids = []
    person_boxes = []

    # -----------------------------
    # Person detection
    # -----------------------------
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2)//2, (y1 + y2)//2

                people_centroids.append([cx, cy])
                person_boxes.append((x1, y1, x2, y2))
                heatmap[cy, cx] += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (57, 255, 20), 3)

    # -----------------------------
    # Cluster detection
    # -----------------------------
    cluster_count = 0
    if len(people_centroids) > 0:
        people_np = np.array(people_centroids)
        clustering = DBSCAN(eps=CLUSTER_DISTANCE, min_samples=CLUSTER_SIZE_THRESHOLD).fit(people_np)
        labels = clustering.labels_
        unique_clusters = set(labels)
        if -1 in unique_clusters:
            unique_clusters.remove(-1)
        cluster_count = len(unique_clusters)

    # -----------------------------
    # Fight overlap detection
    # -----------------------------
    fight_detected = False
    for i in range(len(person_boxes)):
        for j in range(i + 1, len(person_boxes)):
            x1, y1, x2, y2 = person_boxes[i]
            x1b, y1b, x2b, y2b = person_boxes[j]

            if max(x1, x1b) < min(x2, x2b) and max(y1, y1b) < min(y2, y2b):
                fight_detected = True

    # -----------------------------
    # Push detection (stable)
    # -----------------------------
    push_detected = False
    if len(prev_centroids) >= 2 and len(people_centroids) >= 2:
        for (cx1, cy1) in people_centroids:
            for (cx2, cy2) in prev_centroids:
                dist = np.linalg.norm(np.array([cx1, cy1]) - np.array([cx2, cy2]))
                movement = abs(cx1 - cx2) + abs(cy1 - cy2)

                if dist < PUSH_DISTANCE_THRESHOLD and movement > PUSH_SPEED_THRESHOLD:
                    push_detected = True

    prev_centroids = people_centroids.copy()

    # -----------------------------
    # Punch detection (only near another person)
    # -----------------------------
    punch_detected = False
    results_pose = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results_pose.pose_landmarks and len(people_centroids) >= 2:
        lm = results_pose.pose_landmarks.landmark
        wrist = (
            int(lm[mp_pose.PoseLandmark.RIGHT_WRIST].x * w),
            int(lm[mp_pose.PoseLandmark.RIGHT_WRIST].y * h)
        )

        near_person = False
        for (cx, cy) in people_centroids:
            if np.linalg.norm(np.array(wrist) - np.array([cx, cy])) < MIN_INTERACTION_DISTANCE:
                near_person = True
                break

        if prev_wrist is not None and near_person:
            speed = abs(wrist[0] - prev_wrist[0]) + abs(wrist[1] - prev_wrist[1])
            if speed > PUNCH_SPEED_THRESHOLD:
                punch_detected = True

        prev_wrist = wrist

    # -----------------------------
    # Confirm attack across frames + cooldown
    # -----------------------------
    attack_now = fight_detected or push_detected or punch_detected

    if attack_now:
        attack_counter += 1
    else:
        attack_counter = 0

    current_time = time.time()

    if attack_counter >= ATTACK_CONFIRM_FRAMES:
        cv2.putText(frame, "ATTACK DETECTED!", (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        if current_time - last_alarm_time > ALARM_COOLDOWN:
            threading.Thread(target=play_attack_sound).start()
            last_alarm_time = current_time

    # -----------------------------
    # Heatmap visualization
    # -----------------------------
    heatmap_blur = cv2.GaussianBlur(heatmap, (51, 51), 0)
    heatmap_norm = cv2.normalize(heatmap_blur, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.resize(heatmap_color, (w, h))
    overlay = cv2.addWeighted(heatmap_color, 0.6, frame, 0.4, 0)

    # -----------------------------
    # Info display
    # -----------------------------
    cv2.putText(overlay, f"People: {len(people_centroids)}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(overlay, f"Clusters: {cluster_count}", (w - 250, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    if len(people_centroids) >= ALERT_THRESHOLD:
        cv2.putText(overlay, "ALERT: CROWD FORMING", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    if attack_counter >= ATTACK_CONFIRM_FRAMES:
        cv2.putText(overlay, "ALERT: ATTACK ACTIVITY!", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # -----------------------------
    # Show window
    # -----------------------------
    cv2.namedWindow("CrowdHawk AI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CrowdHawk AI", 1280, 720)
    cv2.imshow("CrowdHawk AI", overlay)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

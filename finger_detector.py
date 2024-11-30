import time

import cv2
import mediapipe as mp
import numpy as np

#создаем детектор
handsDetector = mp.solutions.hands.Hands()
cap = cv2.VideoCapture(0)
i = 0
while True:
    i += 1
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break
    imgBGR = np.fliplr(frame)
    # переводим его в формат RGB для распознавания
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
    m, n = imgRGB.shape[:2]
    # Распознаем
    results = handsDetector.process(imgRGB)
    index_tip = (0, 0)
    thumb_tip = (n, m)
    if results.multi_hand_landmarks is not None:
        index_tip = results.multi_hand_landmarks[0].landmark[8]
        index_tip = int(index_tip.x * n), int(index_tip.y * m)
        thumb_tip = results.multi_hand_landmarks[0].landmark[4]
        thumb_tip = int(thumb_tip.x * n), int(thumb_tip.y * m)
    imgRGB.fill(0)
    cv2.circle(imgRGB, index_tip, 5, (255, 0, 0), -1)
    cv2.circle(imgRGB, thumb_tip, 5, (255, 0, 0), -1)
    dis = (thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2
    tap_point = (-1, -1)
    if dis < 2000:
        tap_point = (int((thumb_tip[0] + index_tip[0]) / 2), int((thumb_tip[1] + index_tip[1]) / 2))
        cv2.circle(imgRGB, tap_point, 5, (0, 255, 0), -1)

    # переводим в BGR и показываем результат
    imgBGR = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hands", imgBGR)

# освобождаем ресурсы
handsDetector.close()

import cv2
import mediapipe as mp
import random

from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions.drawing_utils import draw_landmarks

cap = cv2.VideoCapture(0)

# Mock class labels for A-Z, Space, and Delete
gesture_labels = [chr(i) for i in range(65, 91)] + ["Space", "Delete"]

def mock_predict(hand_roi):
    # Simulate a random prediction for demonstration
    return random.choice(gesture_labels)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        bbox = None
        predicted_gesture = None
        if results.multi_hand_landmarks:  # type: ignore[attr-defined]
            for hand_landmarks in results.multi_hand_landmarks:  # type: ignore[attr-defined]
                draw_landmarks(frame, hand_landmarks, list(mp_hands.HAND_CONNECTIONS))
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x_min = int(min(x_coords) * w)
                x_max = int(max(x_coords) * w)
                y_min = int(min(y_coords) * h)
                y_max = int(max(y_coords) * h)
                bbox = (x_min, y_min, x_max, y_max)
                # Draw orange bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 140, 255), 2)
                hand_roi = frame[y_min:y_max, x_min:x_max]
                # Mock prediction
                if hand_roi.size > 0:
                    predicted_gesture = mock_predict(hand_roi)
                    # Draw white filled rectangle for text background
                    text = f'{predicted_gesture}'
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    thickness = 2
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                    rect_x1 = x_min
                    rect_y1 = y_min - text_height - 10 if y_min - text_height - 10 > 0 else y_min + 10
                    rect_x2 = rect_x1 + text_width + 10
                    rect_y2 = rect_y1 + text_height + baseline + 10
                    cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
                    # Draw black text
                    cv2.putText(frame, text, (rect_x1 + 5, rect_y2 - baseline - 5), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        cv2.imshow('ASL Hand Detection', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()

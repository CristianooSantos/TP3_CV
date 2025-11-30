import cv2
import mediapipe as mp

class HandGestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        return result.multi_hand_landmarks, result.multi_handedness

    def finger_extended(self, landmarks, tip_id, pip_id, w, h, hand_label=None, finger_name=None):
        tip = landmarks.landmark[tip_id]
        pip = landmarks.landmark[pip_id]

        if finger_name == "THUMB" and hand_label:
            if hand_label == "Right":
                return (tip.x * w) > (pip.x * w)
            else:
                return (tip.x * w) < (pip.x * w)

        return (tip.y * h) < (pip.y * h)

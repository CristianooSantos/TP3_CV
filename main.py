import cv2
import mediapipe as mp
from audio_engine import AudioEngine
from hand_gesture_recognizer import HandGestureRecognizer
import pygame

class TP3:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.audio = AudioEngine()
        self.rec = HandGestureRecognizer()

        self.fingers = {
            "THUMB": (4, 2),
            "INDEX": (8, 6),
            "MIDDLE": (12, 10),
            "RING": (16, 14),
            "PINKY": (20, 18),
        }

        self.instrument = "Piano"

        self.mp_face = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        # Pastas corretas
        self.instrument_folders = {
            "Piano": "Sons/Piano",
            "Guitarra": "  Somns/Guitarra"
        }

    def draw_finger_text(self, frame, finger_name, note, landmark, w, h):
        x = int(landmark.x * w)
        y = int(landmark.y * h) - 25
        cv2.putText(frame, f"{finger_name}: {note}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def detect_head_lean(self, face_landmarks):
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]

        dx = right_eye.x - left_eye.x
        dy = right_eye.y - left_eye.y
        slope = dy / abs(dx) if abs(dx) > 0.0001 else 0

        if slope > 0.10:
            return "LEFT"
        elif slope < -0.10:
            return "RIGHT"
        return "CENTER"

    def play_note(self, note):
        folder = self.instrument_folders[self.instrument]
        file_path = f"{folder}/{note}_{self.instrument}.wav"
        self.audio.play(self.instrument, note)


    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            face = self.mp_face.process(frame)
            if face.multi_face_landmarks:
                lean_dir = self.detect_head_lean(face.multi_face_landmarks[0])
                if lean_dir == "RIGHT":
                    self.instrument = "Guitarra"
                elif lean_dir == "LEFT":
                    self.instrument = "Piano"

            cv2.putText(frame, self.instrument, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

            hands, handedness = self.rec.process(frame)
            final_frame = frame.copy()

            if hands:
                for idx, hand in enumerate(hands):
                    self.mp_drawing.draw_landmarks(
                        final_frame, hand,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        self.mp_styles.get_default_hand_landmarks_style(),
                        self.mp_styles.get_default_hand_connections_style()
                    )

            if hands:
                for idx, hand in enumerate(hands):
                    label = handedness[idx].classification[0].label

                    if label == "Right":
                        if self.rec.finger_extended(hand, *self.fingers["PINKY"], w, h):
                            self.play_note("DO")
                            self.draw_finger_text(final_frame, "PINKY", "DO", hand.landmark[20], w, h)

                        if self.rec.finger_extended(hand, *self.fingers["RING"], w, h):
                            self.play_note("RE")
                            self.draw_finger_text(final_frame, "RING", "RE", hand.landmark[16], w, h)

                        if self.rec.finger_extended(hand, *self.fingers["MIDDLE"], w, h):
                            self.play_note("MI")
                            self.draw_finger_text(final_frame, "MIDDLE", "MI", hand.landmark[12], w, h)

                        if self.rec.finger_extended(hand, *self.fingers["INDEX"], w, h):
                            self.play_note("FA")
                            self.draw_finger_text(final_frame, "INDEX", "FA", hand.landmark[8], w, h)

                        if self.rec.finger_extended(hand, *self.fingers["THUMB"], w, h, label, "THUMB"):
                            self.play_note("SOL")
                            self.draw_finger_text(final_frame, "THUMB", "SOL", hand.landmark[4], w, h)

                    if label == "Left":
                        if self.rec.finger_extended(hand, *self.fingers["INDEX"], w, h):
                            self.play_note("LA")
                            self.draw_finger_text(final_frame, "INDEX", "LA", hand.landmark[8], w, h)

                        if self.rec.finger_extended(hand, *self.fingers["MIDDLE"], w, h):
                            self.play_note("SI")
                            self.draw_finger_text(final_frame, "MIDDLE", "SI", hand.landmark[12], w, h)

                        if self.rec.finger_extended(hand, *self.fingers["THUMB"], w, h, label, "THUMB"):
                            self.play_note("SOL")
                            self.draw_finger_text(final_frame, "THUMB", "SOL", hand.landmark[4], w, h)

            cv2.imshow("Maestro", final_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()


if __name__ == "__main__":
    app = TP3()
    app.run()

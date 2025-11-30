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

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

    def draw_finger_text(self, frame, finger_name, note, landmark, w, h):
        x = int(landmark.x * w)
        y = int(landmark.y * h) - 25
        cv2.putText(
            frame,
            f"{finger_name}: {note}",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    def run(self):
        print("A correr MVP de gestos. 'q' para sair.")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape


            hands, handedness = self.rec.process(frame)

            final_frame = frame.copy()
            if hands:
                for idx, hand in enumerate(hands):
                    self.mp_drawing.draw_landmarks(
                        final_frame,
                        hand,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        self.mp_styles.get_default_hand_landmarks_style(),
                        self.mp_styles.get_default_hand_connections_style()
                    )

    
            if hands:
                for idx, hand in enumerate(hands):
                    label = handedness[idx].classification[0].label

                
                    if label == "Right":
                        if self.rec.finger_extended(hand, *self.fingers["PINKY"], w, h):
                            self.audio.play("DO")
                            self.draw_finger_text(final_frame, "PINKY", "DO", hand.landmark[20], w, h)

                        if self.rec.finger_extended(hand, *self.fingers["RING"], w, h):
                            self.audio.play("RE")
                            self.draw_finger_text(final_frame, "RING", "RE", hand.landmark[16], w, h)

                        if self.rec.finger_extended(hand, *self.fingers["MIDDLE"], w, h):
                            self.audio.play("MI")
                            self.draw_finger_text(final_frame, "MIDDLE", "MI", hand.landmark[12], w, h)

                        if self.rec.finger_extended(hand, *self.fingers["INDEX"], w, h):
                            self.audio.play("FA")
                            self.draw_finger_text(final_frame, "INDEX", "FA", hand.landmark[8], w, h)

                        if self.rec.finger_extended(hand, *self.fingers["THUMB"], w, h, label, "THUMB"):
                            self.audio.play("SOL")
                            self.draw_finger_text(final_frame, "THUMB", "SOL", hand.landmark[4], w, h)

                
                    if label == "Left":
                        if self.rec.finger_extended(hand, *self.fingers["INDEX"], w, h):
                            self.audio.play("LA")
                            self.draw_finger_text(final_frame, "INDEX", "LA", hand.landmark[8], w, h)

                        if self.rec.finger_extended(hand, *self.fingers["MIDDLE"], w, h):
                            self.audio.play("SI")
                            self.draw_finger_text(final_frame, "MIDDLE", "SI", hand.landmark[12], w, h)

                        if self.rec.finger_extended(hand, *self.fingers["THUMB"], w, h, label, "THUMB"):
                            self.audio.play("SOL")
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

import cv2
import mediapipe as mp
from audio_engine import AudioEngine
from hand_gesture_recognizer import HandGestureRecognizer
from ultralytics import YOLO


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

        # ----- CARROSSEL DE INSTRUMENTOS -----
        self.instruments = ["Piano", "Guitarra", "Objetos"]
        self.instrument_index = 0
        self.instrument = self.instruments[self.instrument_index]
        self.last_lean = "CENTER"  # para não trocar 1000x enquanto a cabeça está inclinada

        # Face mesh (para detetar inclinação da cabeça)
        self.mp_face = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        # YOLO MODEL
        self.yolo = YOLO("yolov8n.pt")

    def detect_head_lean(self, face_landmarks):
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]

        dx = right_eye.x - left_eye.x
        dy = right_eye.y - left_eye.y
        slope = dy / abs(dx) if abs(dx) > 0.0001 else 0

        if slope > 0.35:
            return "LEFT"
        elif slope < -0.35:
            return "RIGHT"
        return "CENTER"

    def play_note(self, note):
        self.audio.play(self.instrument, note)

    def draw_finger_text(self, frame, finger_name, note, landmark, w, h):
        x = int(landmark.x * w)
        y = int(landmark.y * h) - 25
        cv2.putText(frame, f"{finger_name}: {note}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            final_frame = frame.copy()

            # ---------------------------------
            # HEAD TILT / CARROSSEL / MODOS
            # ---------------------------------
            face = self.mp_face.process(frame)
            if face.multi_face_landmarks:
                lean_dir = self.detect_head_lean(face.multi_face_landmarks[0])

                if lean_dir != self.last_lean:
                    if lean_dir == "RIGHT":
                        self.instrument_index = (self.instrument_index + 1) % len(self.instruments)
                        self.instrument = self.instruments[self.instrument_index]
                        print("Modo:", self.instrument)
                    elif lean_dir == "LEFT":
                        self.instrument_index = (self.instrument_index - 1) % len(self.instruments)
                        self.instrument = self.instruments[self.instrument_index]
                        print("Modo:", self.instrument)
                    self.last_lean = lean_dir
            else:
                self.last_lean = "CENTER"

            cv2.putText(final_frame, self.instrument, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

            # -----------------------------------
            # YOLO: APENAS NO MODO "Objetos"
            # -----------------------------------
            if self.instrument == "Objetos":
                yolo_results = self.yolo.predict(frame, conf=0.60, verbose=False)

                for r in yolo_results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        label = r.names[cls]

                        # ignorar pessoas
                        if label.lower() == "person":
                            continue

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(final_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(final_frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # -----------------------------------
            # HAND GESTURES (DESLIGADO EM "Objetos")
            # -----------------------------------
            if self.instrument != "Objetos":
                hands, handedness = self.rec.process(frame)

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

            cv2.imshow("Maestro", final_frame)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = TP3()
    app.run()

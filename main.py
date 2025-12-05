import cv2
import mediapipe as mp
from audio_engine import AudioEngine
from hand_gesture_recognizer import HandGestureRecognizer
from ultralytics import YOLO
import pygame
import os


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

        self.finger_state = {
            "Right": {name: False for name in self.fingers.keys()},
            "Left": {name: False for name in self.fingers.keys()},
        }

        self.instruments = ["Piano", "Guitarra", "Objetos"]
        self.instrument_index = 0
        self.instrument = self.instruments[self.instrument_index]
        self.last_lean = "CENTER"

        self.octave = 0
        self.min_octave = -1
        self.max_octave = 0

        self.octave_high_zone = 0.4
        self.octave_low_zone = 0.6

        self.octave_control_enabled = False

        self.mp_face = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.yolo = YOLO("yolov8n.pt")

        self.object_music = {
            "book": "Sons/Objetos/book.mp3",
            "cup": "Sons/Objetos/copo.mp3",
            "bottle": "Sons/Objetos/garrafa.mp3",
            "laptop": "Sons/Objetos/laptop.mp3",
            "apple": "Sons/Objetos/maca.mp3",
        }

        self.last_object = None
        self.cooldown_frames = 0

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
            2,
        )

    def play_object_music(self, label):
        path = self.object_music.get(label)
        if not path:
            return

        if os.path.exists(path):
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            print(f"🎵 A tocar música para objeto: {label}")

    def reset_finger_state_and_stop_notes(self):
        self.audio.stop_all()
        for hand in self.finger_state:
            for finger in self.finger_state[hand]:
                self.finger_state[hand][finger] = False

    def retrigger_active_notes(self):
        for finger, is_ext in self.finger_state["Right"].items():
            if not is_ext:
                continue

            if finger == "PINKY":
                self.audio.note_on(self.instrument, "DO")
            if finger == "RING":
                self.audio.note_on(self.instrument, "RE")
            if finger == "MIDDLE":
                self.audio.note_on(self.instrument, "MI")
            if finger == "INDEX":
                self.audio.note_on(self.instrument, "FA")
            if finger == "THUMB":
                self.audio.note_on(self.instrument, "SOL")

        for finger, is_ext in self.finger_state["Left"].items():
            if not is_ext:
                continue

            if finger == "INDEX":
                self.audio.note_on(self.instrument, "LA")
            if finger == "MIDDLE":
                self.audio.note_on(self.instrument, "SI")

    def update_octave_from_right_arm(self, hands, handedness):
        if not self.octave_control_enabled:
            return

        right_wrist_y = None

        if hands and handedness:
            for idx, hand in enumerate(hands):
                label = handedness[idx].classification[0].label
                if label == "Right":
                    right_wrist_y = hand.landmark[0].y
                    break

        if right_wrist_y is None:
            return

        new_octave = self.octave

        if right_wrist_y < self.octave_high_zone:
            new_octave = 0
        elif right_wrist_y > self.octave_low_zone:
            new_octave = -1
        else:
            return

        if new_octave != self.octave:
            self.octave = new_octave
            print(f"[Octave] Changed to {self.octave}")

            if hasattr(self.audio, "set_octave"):
                self.audio.set_octave(self.octave)

            self.retrigger_active_notes()

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.audio.update()

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            final_frame = frame.copy()

            face = self.mp_face.process(frame)
            if face.multi_face_landmarks:
                lean_dir = self.detect_head_lean(face.multi_face_landmarks[0])

                if lean_dir != self.last_lean:
                    if lean_dir == "RIGHT":
                        self.instrument_index = (self.instrument_index + 1) % len(
                            self.instruments
                        )
                        self.instrument = self.instruments[self.instrument_index]
                        print("Modo:", self.instrument)
                    elif lean_dir == "LEFT":
                        self.instrument_index = (self.instrument_index - 1) % len(
                            self.instruments
                        )
                        self.instrument = self.instruments[self.instrument_index]
                        print("Modo:", self.instrument)
                    self.last_lean = lean_dir
            else:
                self.last_lean = "CENTER"

            if self.instrument != "Objetos":
                pygame.mixer.music.stop()
                self.last_object = None
                self.cooldown_frames = 0
            else:
                self.reset_finger_state_and_stop_notes()

            cv2.putText(
                final_frame,
                self.instrument,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                3,
            )

            cv2.putText(
                final_frame,
                f"Octave: {self.octave}",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (200, 200, 200),
                2,
            )

            octave_status = "ON" if self.octave_control_enabled else "OFF"
            cv2.putText(
                final_frame,
                f"OctCtl: {octave_status}",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (180, 180, 255),
                2,
            )

            y_high = int(self.octave_high_zone * h)
            y_low = int(self.octave_low_zone * h)
            cv2.line(final_frame, (0, y_high), (w, y_high), (100, 100, 255), 1)
            cv2.line(final_frame, (0, y_low), (w, y_low), (100, 255, 100), 1)

            detected_object = None

            if self.instrument == "Objetos":
                yolo_results = self.yolo.predict(frame, conf=0.60, verbose=False)

                for r in yolo_results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        yolo_label = r.names[cls]

                        if yolo_label.lower() == "person":
                            continue

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(
                            final_frame, (x1, y1), (x2, y2), (0, 255, 255), 2
                        )
                        cv2.putText(
                            final_frame,
                            yolo_label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2,
                        )

                        if yolo_label in self.object_music:
                            detected_object = yolo_label

                if detected_object:
                    if (detected_object != self.last_object) or (
                        self.cooldown_frames == 0
                    ):
                        if not pygame.mixer.music.get_busy():
                            self.play_object_music(detected_object)
                            self.last_object = detected_object
                            self.cooldown_frames = 30

                if self.cooldown_frames > 0:
                    self.cooldown_frames -= 1

            if self.instrument != "Objetos":
                hands, handedness = self.rec.process(frame)

                self.octave_control_enabled = False
                if hands and handedness:
                    for idx, hand in enumerate(hands):
                        hand_label = handedness[idx].classification[0].label
                        if hand_label == "Left":
                            all_ext = True
                            for fname, (tip_id, pip_id) in self.fingers.items():
                                if fname == "THUMB":
                                    ext = self.rec.finger_extended(
                                        hand, tip_id, pip_id, w, h, hand_label, "THUMB"
                                    )
                                else:
                                    ext = self.rec.finger_extended(
                                        hand, tip_id, pip_id, w, h
                                    )
                                if not ext:
                                    all_ext = False
                                    break
                            if all_ext:
                                self.octave_control_enabled = True
                                break

                self.update_octave_from_right_arm(hands, handedness)

                if hands:
                    for idx, hand in enumerate(hands):
                        self.mp_drawing.draw_landmarks(
                            final_frame,
                            hand,
                            mp.solutions.hands.HAND_CONNECTIONS,
                            self.mp_styles.get_default_hand_landmarks_style(),
                            self.mp_styles.get_default_hand_connections_style(),
                        )

                    for idx, hand in enumerate(hands):
                        hand_label = handedness[idx].classification[0].label

                        if hand_label == "Right":
                            prev = self.finger_state["Right"]["PINKY"]
                            is_ext = self.rec.finger_extended(
                                hand, *self.fingers["PINKY"], w, h
                            )
                            if is_ext and not prev:
                                self.audio.note_on(self.instrument, "DO")
                            if not is_ext and prev:
                                self.audio.note_off(self.instrument, "DO")
                            if is_ext:
                                self.draw_finger_text(
                                    final_frame,
                                    "PINKY",
                                    "DO",
                                    hand.landmark[20],
                                    w,
                                    h,
                                )
                            self.finger_state["Right"]["PINKY"] = is_ext

                            prev = self.finger_state["Right"]["RING"]
                            is_ext = self.rec.finger_extended(
                                hand, *self.fingers["RING"], w, h
                            )
                            if is_ext and not prev:
                                self.audio.note_on(self.instrument, "RE")
                            if not is_ext and prev:
                                self.audio.note_off(self.instrument, "RE")
                            if is_ext:
                                self.draw_finger_text(
                                    final_frame,
                                    "RING",
                                    "RE",
                                    hand.landmark[16],
                                    w,
                                    h,
                                )
                            self.finger_state["Right"]["RING"] = is_ext

                            prev = self.finger_state["Right"]["MIDDLE"]
                            is_ext = self.rec.finger_extended(
                                hand, *self.fingers["MIDDLE"], w, h
                            )
                            if is_ext and not prev:
                                self.audio.note_on(self.instrument, "MI")
                            if not is_ext and prev:
                                self.audio.note_off(self.instrument, "MI")
                            if is_ext:
                                self.draw_finger_text(
                                    final_frame,
                                    "MIDDLE",
                                    "MI",
                                    hand.landmark[12],
                                    w,
                                    h,
                                )
                            self.finger_state["Right"]["MIDDLE"] = is_ext

                            prev = self.finger_state["Right"]["INDEX"]
                            is_ext = self.rec.finger_extended(
                                hand, *self.fingers["INDEX"], w, h
                            )
                            if is_ext and not prev:
                                self.audio.note_on(self.instrument, "FA")
                            if not is_ext and prev:
                                self.audio.note_off(self.instrument, "FA")
                            if is_ext:
                                self.draw_finger_text(
                                    final_frame,
                                    "INDEX",
                                    "FA",
                                    hand.landmark[8],
                                    w,
                                    h,
                                )
                            self.finger_state["Right"]["INDEX"] = is_ext

                            prev = self.finger_state["Right"]["THUMB"]
                            is_ext = self.rec.finger_extended(
                                hand,
                                *self.fingers["THUMB"],
                                w,
                                h,
                                hand_label,
                                "THUMB",
                            )
                            if is_ext and not prev:
                                self.audio.note_on(self.instrument, "SOL")
                            if not is_ext and prev:
                                self.audio.note_off(self.instrument, "SOL")
                            if is_ext:
                                self.draw_finger_text(
                                    final_frame,
                                    "THUMB",
                                    "SOL",
                                    hand.landmark[4],
                                    w,
                                    h,
                                )
                            self.finger_state["Right"]["THUMB"] = is_ext

                        if hand_label == "Left":
                            prev = self.finger_state["Left"]["INDEX"]
                            is_ext = self.rec.finger_extended(
                                hand, *self.fingers["INDEX"], w, h
                            )
                            if is_ext and not prev:
                                self.audio.note_on(self.instrument, "LA")
                            if not is_ext and prev:
                                self.audio.note_off(self.instrument, "LA")
                            if is_ext:
                                self.draw_finger_text(
                                    final_frame,
                                    "INDEX",
                                    "LA",
                                    hand.landmark[8],
                                    w,
                                    h,
                                )
                            self.finger_state["Left"]["INDEX"] = is_ext

                            prev = self.finger_state["Left"]["MIDDLE"]
                            is_ext = self.rec.finger_extended(
                                hand, *self.fingers["MIDDLE"], w, h
                            )
                            if is_ext and not prev:
                                self.audio.note_on(self.instrument, "SI")
                            if not is_ext and prev:
                                self.audio.note_off(self.instrument, "SI")
                            if is_ext:
                                self.draw_finger_text(
                                    final_frame,
                                    "MIDDLE",
                                    "SI",
                                    hand.landmark[12],
                                    w,
                                    h,
                                )
                            self.finger_state["Left"]["MIDDLE"] = is_ext
                else:
                    self.reset_finger_state_and_stop_notes()

            cv2.imshow("Maestro", final_frame)
            if cv2.waitKey(50) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()


if __name__ == "__main__":
    app = TP3()
    app.run()

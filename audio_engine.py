import pygame
import time

class AudioEngine:
    def __init__(self):
        pygame.mixer.init()

        self.piano = {
            "DO": pygame.mixer.Sound("Sons/Piano/E5.wav"),
            "RE": pygame.mixer.Sound("Sons/Piano/F5.wav"),
            "MI": pygame.mixer.Sound("Sons/Piano/G5.wav"),
            "FA": pygame.mixer.Sound("Sons/Piano/A5.wav"),
            "SOL": pygame.mixer.Sound("Sons/Piano/B5.wav"),
            "LA": pygame.mixer.Sound("Sons/Piano/C5.wav"),
            "SI": pygame.mixer.Sound("Sons/Piano/D5.wav"),
        }

        self.guitarra = {
            "DO": pygame.mixer.Sound("Sons/Guitarra/DO_Guitarra.wav"),
            "RE": pygame.mixer.Sound("Sons/Guitarra/RE_Guitarra.wav"),
            "MI": pygame.mixer.Sound("Sons/Guitarra/MI_Guitarra.wav"),
            "FA": pygame.mixer.Sound("Sons/Guitarra/FA_Guitarra.wav"),
            "SOL": pygame.mixer.Sound("Sons/Guitarra/SOL_Guitarra.wav"),
            "LA": pygame.mixer.Sound("Sons/Guitarra/LA_Guitarra.wav"),
            "SI": pygame.mixer.Sound("Sons/Guitarra/LA_Guitarra.wav"),
        }

        self.active_notes = {}

    def note_on(self, instrument, note):
        snd = self._get_sound(instrument, note)
        if snd:
            snd.play(loops=-1)
            key = (instrument, note)
            if key in self.active_notes:
                del self.active_notes[key] 

    def note_off(self, instrument, note, release_seconds=1):
        stop_time = time.time() + release_seconds
        self.active_notes[(instrument, note)] = stop_time

    def update(self):
        """Must be called every frame to stop released notes."""
        now = time.time()
        for key, stop_time in list(self.active_notes.items()):
            if now >= stop_time:
                instrument, note = key
                snd = self._get_sound(instrument, note)
                if snd:
                    snd.stop()
                del self.active_notes[key]

    def stop_all(self):
        for snd in self.piano.values():
            snd.stop()
        for snd in self.guitarra.values():
            snd.stop()
        self.active_notes.clear()

    def _get_sound(self, instrument, note):
        if instrument == "Piano":
            return self.piano.get(note)
        elif instrument == "Guitarra":
            return self.guitarra.get(note)
        return None

import pygame
import time


class AudioEngine:
    def __init__(self):
        pygame.mixer.init()

        piano_octave_0 = {
            "DO": "Sons/Piano/E5.wav",
            "RE": "Sons/Piano/F5.wav",
            "MI": "Sons/Piano/G5.wav",
            "FA": "Sons/Piano/A5.wav",
            "SOL": "Sons/Piano/B5.wav",
            "LA": "Sons/Piano/C5.wav",
            "SI": "Sons/Piano/D5.wav",
        }

        piano_octave_minus_1 = {
            "DO": "Sons/Piano/E4.wav",
            "RE": "Sons/Piano/F4.wav",
            "MI": "Sons/Piano/G4.wav",
            "FA": "Sons/Piano/A4.wav",
            "SOL": "Sons/Piano/B4.wav",
            "LA": "Sons/Piano/C4.wav",
            "SI": "Sons/Piano/D4.wav",
        }

        guitarra_octave_0 = {
            "DO": "Sons/Guitarra/DO_Guitarra.wav",
            "RE": "Sons/Guitarra/RE_Guitarra.wav",
            "MI": "Sons/Guitarra/MI_Guitarra.wav",
            "FA": "Sons/Guitarra/FA_Guitarra.wav",
            "SOL": "Sons/Guitarra/SOL_Guitarra.wav",
            "LA": "Sons/Guitarra/LA_Guitarra.wav",
            "SI": "Sons/Guitarra/LA_Guitarra.wav", 
        }

        self.piano_samples = {
            0: {n: pygame.mixer.Sound(path) for n, path in piano_octave_0.items()},
            -1: {n: pygame.mixer.Sound(path) for n, path in piano_octave_minus_1.items()},
        }

        self.guitarra_samples = {
            0: {n: pygame.mixer.Sound(path) for n, path in guitarra_octave_0.items()},
        }

        self.current_octave = 0

        self.active_notes = {}


    def _get_bank(self, instrument):
        if instrument == "Piano":
            return self.piano_samples
        elif instrument == "Guitarra":
            return self.guitarra_samples
        return None

    def _get_sound(self, instrument, note, octave=None):
        if octave is None:
            octave = self.current_octave

        bank_by_octave = self._get_bank(instrument)
        if not bank_by_octave:
            return None

        bank = bank_by_octave.get(octave) or bank_by_octave.get(0)
        if not bank:
            return None

        return bank.get(note)


    def note_on(self, instrument, note):
        """Start a note (sustained until note_off + release time)."""
        snd = self._get_sound(instrument, note)
        if snd:
            snd.play(loops=-1)
            key = (instrument, note, self.current_octave)
            if key in self.active_notes:
                del self.active_notes[key]

    def note_off(self, instrument, note, release_seconds=1):
        """Schedule this note to stop in `release_seconds` seconds."""
        key = (instrument, note, self.current_octave)
        stop_time = time.time() + release_seconds
        self.active_notes[key] = stop_time

    def update(self):
        """Called every frame to stop notes whose release time is over."""
        now = time.time()
        for key, stop_time in list(self.active_notes.items()):
            if now >= stop_time:
                instrument, note, octave = key
                snd = self._get_sound(instrument, note, octave)
                if snd:
                    snd.stop()
                del self.active_notes[key]

    def stop_all(self):
        """Stop all notes immediately."""
        for bank in self.piano_samples.values():
            for snd in bank.values():
                snd.stop()

        for bank in self.guitarra_samples.values():
            for snd in bank.values():
                snd.stop()

        self.active_notes.clear()

    def set_octave(self, octave: int):
        """Called when user moves the right arm up/down."""
        self.current_octave = octave
        print(f"[AudioEngine] Octave set to {octave}")

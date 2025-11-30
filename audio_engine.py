import pygame

class AudioEngine:
    def __init__(self):
        pygame.mixer.init()
        self.sounds = {
            "DO": pygame.mixer.Sound("Sons/Do6-102822.wav"),
            "RE": pygame.mixer.Sound("Sons/Re6.wav"),
            "MI": pygame.mixer.Sound("Sons/Mi6-82016.wav"),
            "FA": pygame.mixer.Sound("Sons/Fa6.wav"),
            "SOL": pygame.mixer.Sound("Sons/Sol6-82013.wav"),
            "LA": pygame.mixer.Sound("Sons/La6-102820.wav"),
            "SI": pygame.mixer.Sound("Sons/Si6-82017.wav"),
        }

    def play(self, key):
        if key in self.sounds:
            self.sounds[key].play()

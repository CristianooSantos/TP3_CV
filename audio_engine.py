import pygame

class AudioEngine:
    def __init__(self):
        pygame.mixer.init()

        self.piano = {
            "DO": pygame.mixer.Sound("Sons/Piano/Do6-102822.wav"),
            "RE": pygame.mixer.Sound("Sons/Piano/Re6.wav"),
            "MI": pygame.mixer.Sound("Sons/Piano/Mi6-82016.wav"),
            "FA": pygame.mixer.Sound("Sons/Piano/Fa6.wav"),
            "SOL": pygame.mixer.Sound("Sons/Piano/Sol6-82013.wav"),
            "LA": pygame.mixer.Sound("Sons/Piano/La6-102820.wav"),
            "SI": pygame.mixer.Sound("Sons/Piano/Si6-82017.wav"),
        }

        self.guitarra = {
            "DO": pygame.mixer.Sound("Sons/Guitarra/DO_Guitarra.wav"),
            "RE": pygame.mixer.Sound("Sons/Guitarra/RE_Guitarra.wav"),
            "MI": pygame.mixer.Sound("Sons/Guitarra/MI_Guitarra.wav"),
            "FA": pygame.mixer.Sound("Sons/Guitarra/FA_Guitarra.wav"),
            "SOL": pygame.mixer.Sound("Sons/Guitarra/SOL_Guitarra.wav"),
            "LA": pygame.mixer.Sound("Sons/Guitarra/LA_Guitarra.wav"),
            "SI": pygame.mixer.Sound("Sons/Guitarra/LA_Guitarra.wav"), # Falta Encontrar a nota SI no site não achei
        }

    def play(self, instrument, note):
        if instrument == "Piano":
            self.piano[note].play()
        elif instrument == "Guitarra":
            self.guitarra[note].play()

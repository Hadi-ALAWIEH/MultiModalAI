import pyttsx3
import speech_recognition as sr

class VoiceAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def listen(self):
        with sr.Microphone() as source:
            print("Listening...")
            try:
                audio = self.recognizer.listen(source)
                return self.recognizer.recognize_google(audio).lower()
            except sr.UnknownValueError:
                return None
            except sr.RequestError as e:
                print(f"Speech recognition error: {e}")
                return None



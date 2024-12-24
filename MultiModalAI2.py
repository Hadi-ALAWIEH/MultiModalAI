import cv2
from ultralytics import YOLO
from vosk import Model, KaldiRecognizer
import pyaudio
import pyttsx3
import os
import threading
import sys

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()

# Load YOLOv8 Model
model = YOLO("./yolo-Weights/yolov8n.pt")
print(model.names)

# Load Vosk Model for Offline Speech Recognition
current_dir = os.getcwd()
model_path = os.path.join(current_dir, 'vosk', 'vosk-model-small-en-us-0.15')
vosk_model = Model(model_path)
recognizer = KaldiRecognizer(vosk_model, 16000)

# Global variable to indicate detected commands across threads
command_detected = None
command_lock = threading.Lock()
program_running = True  # Control variable to stop the detection loop

# Function to listen for a voice command in a separate thread
def listen_for_command_offline():
    global command_detected, program_running

    mic = pyaudio.PyAudio()
    stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
    stream.start_stream()

    print("Listening for command...")

    while True:
        data = stream.read(4096, exception_on_overflow=False)

        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            command = eval(result).get('text', '').lower()

            with command_lock:
                command_detected = command

                if "goodbye" in command:
                    engine.say("Have a good day")
                    engine.runAndWait()
                    program_running = False
                    sys.exit()

# Function to speak detected objects (run TTS in a separate thread)
def speak_objects(detected_objects, command):
    def speak():
        global program_running

        # Define actions for specific voice commands
        actions = {
            "hello": lambda objs: engine.say("Hey there Hadi, how are you doing?"),
            "i am doing fine and you" : lambda objs: engine.say("I am doing fine as well, let me know if you need anything!"),
            "hi": lambda objs: engine.say("Hey, can I help you today?"),
            "hey": lambda objs: engine.say("Greetings, hope you're having a splendid day!"),
            "what do you see": lambda objs: engine.say("I see the following objects: " + ", ".join(objs)),
            "who are you": lambda objs: engine.say("I am an AI model trained to detect objects in a webcam feed."),
            "who created you": lambda objs: engine.say("My creator is Hadi Alawieh."),
            "see you": lambda objs: (
                engine.say("Have a good day."),
                setattr(sys.modules['__main__'], "program_running", False)
            ),
            "party": lambda objs: engine.say("For a party event, you should wear something stylish and elegant, like a dress or a chic suit."),
            "casual": lambda objs: engine.say("For a casual day, go for comfortable jeans and a t-shirt."),
            "formal": lambda objs: engine.say("For formal events, wear a well-fitted suit or an elegant dress."),
            "workout": lambda objs: engine.say("For workouts, wear athletic clothing like gym shorts and sneakers.")
        }

        action = actions.get(command)
        if action:
            action(detected_objects)

        engine.runAndWait()

    threading.Thread(target=speak).start()




# Function to handle webcam feed and object detection
def detect_objects():
    global command_detected, program_running

    cap = cv2.VideoCapture(0)

    # Define Rebecca Purple color in BGR format (OpenCV uses BGR)
    rebecca_purple = (102, 58, 183)  # BGR color for Rebecca Purple

    while program_running:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.5, show=False)

        detected_objects = []
        for result in results:
            for box in result.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = list(map(int, box[:4])) + [box[4], int(box[5])]
                label = model.names[cls]
                detected_objects.append(label)

                # Draw rectangle around detected object
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, rebecca_purple, 2)

        cv2.imshow("Object Detection", frame)

        with command_lock:
            if command_detected:
                speak_objects(detected_objects, command_detected)
                command_detected = None  # Reset the command

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start listening for commands in a separate thread
command_thread = threading.Thread(target=listen_for_command_offline, daemon=True)
command_thread.start()

# Start object detection
if __name__ == "__main__":
    detect_objects()

from ultralytics import YOLO
import cv2
import math
import threading
from queue import Queue
import time
import pyttsx3  # Import text-to-speech library
import speech_recognition as sr  # Import speech recognition library

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load YOLO model
model = YOLO("../yolo-Weights/yolov8s.pt")

# Object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]
'''
Those are the classes that the yolov8n.pt model is trained on
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
couch
potted plant
bed
dining table
toilet
tv
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
book
clock
vase
scissors
teddy bear
hair drier
toothbrush
'''

# Global variable to store detected objects
detected_objects = []

# Initialize pyttsx3 engine
engine = pyttsx3.init()

# Function for object detection
# Global queue for communication
detection_queue = Queue()


# Function for object detection (updated to use the queue)
def detect_objects():
    listen_for_commands()
    global detected_objects
    start_time = time.time()  # Record the start time
    yield_interval = 5  # Time (in seconds) after which the thread yields control

    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time

        success, img = cap.read()
        if not success:
            continue

        results = model(img, stream=True)
        frame_objects = []

        # Process each detected object
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                confidence = math.ceil((box.conf[0] * 100)) / 100

                cls = int(box.cls[0])
                frame_objects.append(classNames[cls])

                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

        detection_queue.put(list(set(frame_objects)))

        # Show elapsed time
        if elapsed_time > yield_interval:
            print(f"Yielding control after {yield_interval} seconds")
            start_time = time.time()  # Reset the timer

            # time.sleep(0.1)  # Allow other threads to regain control
        cv2.imshow('Webcam', img)

        if cv2.waitKey(1) == ord('q'):
            break



# Function for speech recognition and handling user queries (updated to use the queue)
def listen_for_commands():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        while True:
            try:
                print("Listening for commands...")
                audio = recognizer.listen(source, timeout=60, phrase_time_limit=10)
                question = recognizer.recognize_google(audio).lower()
                print(f"Detected speech: {question}")

                if 'what' in question:
                    if not detection_queue.empty():
                        detected_objects_inner = detection_queue.get()
                        if detected_objects_inner:
                            response = f"I see: {', '.join(detected_objects_inner)}"
                        else:
                            response = "I don't see anything right now."
                        print(response)
                        engine.say(response)
                        engine.runAndWait();
                        # engine.iterate()  # Non-blocking
                    else:
                        print("No objects detected yet.")
                else:
                    print("Unrecognized question format.")
            except sr.UnknownValueError:
                print("Could not understand the audio.")
            except sr.WaitTimeoutError:
                print("Listening timeout.")
            except Exception as e:
                print(f"Error: {e}")


# Start the object detection thread
detection_thread = threading.Thread(target=detect_objects, daemon=True)
detection_thread.start()


# # Start the speech recognition thread
speech_thread = threading.Thread(target=listen_for_commands, daemon=True)
speech_thread.start()

# Keep the main program running
while True:
    time.sleep(1)  # This keeps the main thread alive

# Clean up on exit
cap.release()
cv2.destroyAllWindows()
from ultralytics import YOLO
import cv2
import math
import threading
from queue import Queue
import time
import pyttsx3  # Import text-to-speech library
import speech_recognition as sr  # Import speech recognition library


# # Initialize webcam
# cap : cv2.VideoCapture = cv2.VideoCapture(0) # here 0 means the index of the webcam we are using, since we only have a default laptop webcam we pass in 0
# cap.set(3, 320)
# cap.set(4, 240)

# Load YOLO model trained on clothing dataset
model : YOLO  = YOLO("../yolo-Weights/yolov8n.pt")  # Use your custom-trained YOLO model here


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
modelNames  =  model.names



# # Clothing classes and event compatibility matrix
# classNames = ["shirt", "pants", "dress", "skirt", "shoes", "tie", "blazer", "gown"]
# compatibility_matrix = {
#     "wedding": {"gown": 5, "blazer": 4, "tie": 3},
#     "casual": {"shirt": 5, "pants": 4, "shoes": 3},
#     "formal": {"blazer": 5, "tie": 4, "shirt": 3}
# }
#
# # Global variable to store detected objects
# detected_objects = []
#
# # Initialize pyttsx3 engine
# engine = pyttsx3.init()
#
# # Global queue for communication
# detection_queue = Queue()
#
#
# # Function for object detection
# def detect_objects():
#     global detected_objects
#     while True:
#         success, img = cap.read()
#         results = model(img, stream=True)
#
#         # Reset the detected objects for the current frame
#         frame_objects = []
#
#         # Process each detected object
#         for r in results:
#             boxes = r.boxes
#             for box in boxes:
#                 # Bounding box coordinates
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#
#                 # Draw bounding box on image
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
#
#                 # Confidence score
#                 confidence = math.ceil((box.conf[0] * 100)) / 100
#
#                 # Class name
#                 cls = int(box.cls[0])
#                 frame_objects.append(classNames[cls])  # Add detected class to list
#
#                 # Display class name on the image
#                 org = [x1, y1]
#                 font = cv2.FONT_HERSHEY_SIMPLEX
#                 fontScale = 1
#                 color = (255, 0, 0)
#                 thickness = 2
#                 cv2.putText(img, f"{classNames[cls]} {confidence}", org, font, fontScale, color, thickness)
#
#         # Add the objects detected in the current frame to the queue
#         detection_queue.put(list(set(frame_objects)))
#
#         # Show the frame with detection
#         cv2.imshow('Webcam', img)
#
#         # Exit condition
#         if cv2.waitKey(1) == ord('q'):
#             break
#
#
# # Function for recommending clothing for an event
# def recommend_clothing(event, detected_items):
#     if event not in compatibility_matrix:
#         return "I don't have recommendations for this event."
#
#     event_compatibility = compatibility_matrix[event]
#     recommendations = {}
#
#     for item in detected_items:
#         if item in event_compatibility:
#             recommendations[item] = event_compatibility[item]
#
#     if recommendations:
#         best_match = max(recommendations, key=recommendations.get)
#         return f"The best match for a {event} is {best_match}."
#     else:
#         return f"I couldn't find a suitable item for {event}."
#
#
# # Function for speech recognition and handling user queries
# def listen_for_commands():
#     recognizer = sr.Recognizer()
#
#     while True:
#         with sr.Microphone() as source:
#             print("Say an event (e.g., 'wedding', 'casual', 'formal') to get recommendations.")
#             recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
#             audio = recognizer.listen(source)  # Listen for speech
#
#             try:
#                 # Recognize the speech
#                 query = recognizer.recognize_google(audio).lower()
#                 print(f"You said: {query}")
#
#                 if query in compatibility_matrix:
#                     if not detection_queue.empty():
#                         detected_objects = detection_queue.get()
#                         response = recommend_clothing(query, detected_objects)
#                         print(response)
#                         engine.say(response)
#                         engine.runAndWait()
#                     else:
#                         print("No objects detected yet.")
#                 else:
#                     print("Please say an event like 'wedding', 'casual', or 'formal'.")
#
#             except sr.UnknownValueError:
#                 print("Sorry, I couldn't understand that. Please try again.")
#             except sr.RequestError:
#                 print("Sorry, there was an error with the speech service.")
#
#         time.sleep(0.1)  # Ensure smooth microphone operations
#
#
# # Start the object detection thread
# detection_thread = threading.Thread(target=detect_objects, daemon=True)
# detection_thread.start()
#
# # Start the speech recognition thread
# speech_thread = threading.Thread(target=listen_for_commands, daemon=True)
# speech_thread.start()
#
# # Keep the main program running
# while True:
#     time.sleep(1)  # This keeps the main thread alive
#
# # Clean up on exit
# cap.release()
# cv2.destroyAllWindows()
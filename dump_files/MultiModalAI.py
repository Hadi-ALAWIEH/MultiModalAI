# import cv2
# from ultralytics import YOLO
# from vosk import Model, KaldiRecognizer
# import pyaudio
# import pyttsx3
# import os
# import threading
# import sys
#
# # Initialize Text-to-Speech Engine
# engine = pyttsx3.init()
#
# # Load YOLOv8 Model
# model = YOLO("./yolo-Weights/yolov8n.pt")
#
# # Load Vosk Model for Offline Speech Recognition
# current_dir = os.getcwd()
# model_path = os.path.join(current_dir, 'vosk', 'vosk-model-small-en-us-0.15')
# vosk_model = Model(model_path)
# recognizer = KaldiRecognizer(vosk_model, 16000)
#
# # Global variable to indicate detected commands across threads
# command_detected = None
# command_lock = threading.Lock()
# program_running = True  # Control variable to stop the detection loop
#
# # Function to listen for a voice command in a separate thread
# def listen_for_command_offline():
#     global command_detected, program_running
#
#     mic = pyaudio.PyAudio()
#     stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
#     stream.start_stream()
#
#     print("Listening for command...")
#
#     while True:
#         data = stream.read(4096, exception_on_overflow=False)
#
#         if recognizer.AcceptWaveform(data):
#             result = recognizer.Result()
#             command = eval(result).get('text', '').lower()
#
#             with command_lock:
#                 command_detected = command
#
#                 if "goodbye" in command:
#                     engine.say("Have a good day")
#                     engine.runAndWait()
#
#                     # Stop the detection loop
#                     global program_running
#                     program_running = False
#
#                     # Terminate the program
#                     sys.exit()
#
# # Function to speak detected objects (run TTS in a separate thread)
# def speak_objects(objects):
#     def speak():
#         if objects:
#             message = "I see the following objects: " + ", ".join(objects)
#             print(message)
#             engine.say(message)
#             engine.runAndWait()
#
#     # Run TTS in a separate thread to prevent blocking
#     threading.Thread(target=speak).start()
#
# # Function to handle webcam feed and object detection
# def detect_objects():
#     global command_detected, program_running
#
#     cap = cv2.VideoCapture(0)
#
#     # Define Rebecca Purple color in BGR format (OpenCV uses BGR)
#     rebecca_purple = (102, 58, 183)  # BGR color for Rebecca Purple
#
#     while program_running:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         results = model.predict(source=frame, conf=0.5, show=False)
#
#         detected_objects = []
#         for result in results:
#             for box in result.boxes.data.tolist():
#                 x1, y1, x2, y2, conf, cls = list(map(int, box[:4])) + [box[4], int(box[5])]
#                 label = model.names[cls]
#                 detected_objects.append(label)
#
#                 # Draw rectangle around detected object
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
#                 # Put the label text with Rebecca Purple color
#                 cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, rebecca_purple, 2)
#
#         cv2.imshow("Object Detection", frame)
#
#         with command_lock:
#             if command_detected == "what do you see":
#                 speak_objects(detected_objects)
#                 command_detected = None  # Reset the command
#             else:
#                 if command_detected == "who are you":
#                     engine.say("I am an AI model trained to detect objects in a webcam feed.")
#                     engine.runAndWait()
#                     command_detected = None  # Reset the command
#                 # add another condition
#                 else:
#                     if command_detected == "who created you":
#                         engine.say("My creator is a dumb guy called Hadi Alawieh, its so funny how much of a donkey he is.")
#                         engine.runAndWait()
#                         command_detected = None  # Reset the command
#                     else:
#                         if command_detected == "see you":
#                             engine.say("Have a good day")
#                             engine.runAndWait()
#                             command_detected = None  # Reset the command
#                             program_running = False  # Stop the detection loop
#                             break
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# # Start listening for commands in a separate thread
# command_thread = threading.Thread(target=listen_for_command_offline, daemon=True)
# command_thread.start()
#
# # Start object detection
# if __name__ == "__main__":
#     detect_objects()

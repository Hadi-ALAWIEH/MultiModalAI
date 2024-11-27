import cv2
from object_detection_module import ObjectDetector
from speech_recognition_module import VoiceAssistant

# Initialize components
detector = ObjectDetector("yolo-Weights/yolov8n.pt")
assistant = VoiceAssistant()

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Object detection
    objects, annotated_frame = detector.detect(frame)
    if objects:
        assistant.speak(f"I see {', '.join(objects)}")
        print(f"Detected objects: {', '.join(objects)}")

    # Display webcam feed
    cv2.imshow("Webcam", annotated_frame)

    # Voice commands
    command = assistant.listen()
    if command:
        print(f"Command: {command}")
        if "stop" in command:
            assistant.speak("Goodbye!")
            break
        elif "what do you see" in command:
            if objects:
                assistant.speak(f"I see {', '.join(objects)}")
            else:
                assistant.speak("I don't see anything at the moment.")

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()






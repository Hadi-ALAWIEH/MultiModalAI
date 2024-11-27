import cv2
from object_detection_module import ObjectDetector
from speech_recognition_module import VoiceAssistant

# Event-to-clothing mapping
event_clothing_map = {
    "formal": ["tie", "suitcase", "blazer"],
    "casual": ["t-shirt", "jeans", "hoodie"],
    "sports": ["sports ball", "sneakers", "jersey"]
}

def suggest_clothing(event, detected_objects):
    if not detected_objects:
        return "I don't see any clothing options. Please try again."
    for obj in detected_objects:
        if obj in event_clothing_map.get(event, []):
            return f"I suggest you wear the {obj} for a {event} event."
    return f"I don't see any suitable options for a {event} event."

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
        elif "suggest" in command:
            for event in event_clothing_map.keys():
                if event in command:
                    suggestion = suggest_clothing(event, objects)
                    assistant.speak(suggestion)
                    print(suggestion)
                    break
            else:
                assistant.speak("I couldn't understand the event type. Please try again.")

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

import cv2
import math
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                            "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                            "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                            "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
                            "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote",
                            "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                            "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    def detect(self, img):
        results = self.model(img, stream=True)
        detected_objects = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                confidence = math.ceil((box.conf[0] * 100)) / 100
                detected_objects.append(self.class_names[cls])

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(img, f"{self.class_names[cls]} ({confidence}%)", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return detected_objects, img

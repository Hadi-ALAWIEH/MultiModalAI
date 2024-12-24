# import cv2
# from object_detection_module import ObjectDetector
#
# # Initialize ObjectDetector with the path to the YOLO model
# model_path = "/yolo-Weights/yolov8n.pt"  # Replace with the correct model path if different
# detector = ObjectDetector(model_path)
#
# # Read an image for testing
# img_path = "Tesla wp 3.jpg"  # Replace with your test image path
# img = cv2.imread(img_path)
#
# if img is None:
#     print("Image not found. Please check the path.")
# else:
#     # Get original dimensions of the image
#     height, width = img.shape[:2]
#
#     # Define the target width (smaller than the original size)
#     target_width = 800  # You can change this to your desired width
#
#     # Calculate the scaling factor to maintain the aspect ratio
#     scaling_factor = target_width / width
#     target_height = int(height * scaling_factor)
#
#     # Resize the image while maintaining aspect ratio
#     img_resized = cv2.resize(img, (target_width, target_height))
#
#     # Perform detection on the resized image
#     detected_objects, processed_img = detector.detect(img_resized)
#
#     # Print detected objects
#     print("Detected Objects:")
#     for obj in detected_objects:
#         print(f"- {obj}")
#
#     # Display the processed image with detections
#     cv2.imshow("Detections", processed_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

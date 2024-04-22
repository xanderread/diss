import cv2
from ultralytics import YOLO

def nothing(x):
    pass

# we use the ultralytics YOLO model framework to load the onnx model
model = YOLO('best_model.onnx') 

# Constants
RTMP_IP = "rtmp://127.0.0.1/live/sar"
EXIT_KEY = 'q'

# Setup video capture
cap = cv2.VideoCapture(RTMP_IP)
if not cap.isOpened():
    print("Error opening video stream or file")

# Setup window
cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

# Add a slider for confidence threshold
confidence_slider = 'Confidence Threshold'
cv2.createTrackbar(confidence_slider, 'Video', 10, 100, nothing)  # Slider from 0.1 to 1.0, starting at 0.3

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Frame read failed")
        break

    # Get current positions of the slider
    conf = cv2.getTrackbarPos(confidence_slider, 'Video') / 100.0

    # Run detection model
    results = model(frame, conf=conf, device='mps')
    annotated_frame = results[0].plot()

    # Measure FPS
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Overlay FPS on the video
    cv2.putText(annotated_frame, f"Stream FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated_frame, f"Confidence: {conf:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Video', annotated_frame)

    # Press 'q' on keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord(EXIT_KEY):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()

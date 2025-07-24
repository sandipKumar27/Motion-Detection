import cv2
import time

# Initialize video capture (webcam)
cap = cv2.VideoCapture(0)

# Set a pause to allow the camera to warm up
time.sleep(2)

# Initialize first frame for comparison
first_frame = None

print("📹 Motion detection started. Press 'q' to quit.")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    text = "No motion"

    # Convert to grayscale and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Store the first frame as reference
    if first_frame is None:
        first_frame = gray
        continue

    # Compare current frame to first frame
    frame_delta = cv2.absdiff(first_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilate the thresholded image to fill in holes
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles for motion
    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Motion Detected"

    # Show status and frames
    cv2.putText(frame, f"Status: {text}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255) if text == "Motion Detected" else (255, 255, 255), 2)

    cv2.imshow("Live Feed", frame)
    cv2.imshow("Threshold", thresh)
    cv2.imshow("Frame Delta", frame_delta)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

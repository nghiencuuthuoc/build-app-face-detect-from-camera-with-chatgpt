import cv2
import numpy as np
from PIL import Image, ImageDraw
import mediapipe as mp

# Initialize mediapipe for face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize face detection model
with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                # Get the bounding box of the detected face
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                
                # Extend the bounding box to include the shoulders
                extended_y = max(0, y - int(h / 2))  # Adjusting the upper bound
                extended_h = h + int(h / 2)  # Including the shoulders
                
                # Determine the size for the square crop
                square_size = max(w, extended_h)  # Make the crop square based on the larger dimension

                # Adjust the x coordinate to center the square crop
                extended_x = max(0, x - (square_size - w) // 2)
                
                # Crop the square region from the frame (shoulders to head)
                body_img = frame[extended_y:extended_y+square_size, extended_x:extended_x+square_size]
                
                # Convert the cropped image to a PIL Image
                pil_img = Image.fromarray(cv2.cvtColor(body_img, cv2.COLOR_BGR2RGB))

                # Create a circular mask
                mask = Image.new('L', pil_img.size, 0)
                draw = ImageDraw.Draw(mask)

                # Adjust the size of the circle to fit the cropped square
                circle_radius = int(pil_img.size[0] / 2)  # Radius is half the size of the square
                center_x, center_y = pil_img.size[0] // 2, pil_img.size[1] // 2

                # Draw a circle
                draw.ellipse((center_x - circle_radius, center_y - circle_radius,
                              center_x + circle_radius, center_y + circle_radius), fill=255)

                # Apply the circular mask
                pil_img.putalpha(mask)

                # Create a transparent background image to paste the circular image
                circle_frame = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
                circle_frame.paste(pil_img, (0, 0), mask=pil_img)

                # Convert back to OpenCV image
                final_img = cv2.cvtColor(np.array(circle_frame), cv2.COLOR_RGBA2BGRA)

                # Display the final image with circular crop
                cv2.imshow('Body in Square and Circle', final_img)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

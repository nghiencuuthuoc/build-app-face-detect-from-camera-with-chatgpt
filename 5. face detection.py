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
                
                # Crop the face region from the frame
                face_img = frame[y:y+h, x:x+w]
                
                # Convert the cropped face to a PIL Image
                pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

                # Create a circular mask
                mask = Image.new('L', pil_img.size, 0)
                draw = ImageDraw.Draw(mask)

                # Adjust the size of the circle to fit the face better
                circle_radius = int(max(pil_img.size) * 1.2)  # 1.2 factor to make the circle bigger
                center_x, center_y = pil_img.size[0] // 2, pil_img.size[1] // 2

                # Draw a larger circle
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
                cv2.imshow('Face in Circle', final_img)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

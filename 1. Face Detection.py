import cv2
import tkinter as tk
from PIL import Image, ImageTk

# Load the pre-trained Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set up the Tkinter window
window = tk.Tk()
window.title("Face Detection")

# Create a label to display the video frames
label = tk.Label(window)
label.pack()

# Start the webcam
cap = cv2.VideoCapture(0)

def show_frame():
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        return

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Crop the first detected face
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_frame = frame[y:y + h, x:x + w]
    else:
        face_frame = frame

    # Convert the frame to Image format for Tkinter
    face_image = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(face_image)
    img = ImageTk.PhotoImage(img)

    # Update the label with the new frame
    label.img = img  # Keep a reference to the image to avoid garbage collection
    label.config(image=img)

    # Call the function again to display the next frame
    label.after(10, show_frame)

# Start the video feed
show_frame()

# Start the Tkinter event loop
window.mainloop()

# Release the camera when done
cap.release()

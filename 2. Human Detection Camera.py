import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# Initialize Tkinter
root = tk.Tk()
root.title("Human Detection Camera")

# Create a label widget to display the camera feed
label = tk.Label(root)
label.pack()

# Load the pre-trained HOG + SVM model for human detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Initialize the webcam (0 is typically the default camera)
cap = cv2.VideoCapture(0)

# Function to process the frame and display human detection
def show_frame():
    ret, frame = cap.read()

    if ret:
        # Resize the frame for better performance
        frame_resized = cv2.resize(frame, (640, 480))

        # Convert the frame to grayscale (required for detection)
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # Detect people in the frame
        boxes, weights = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)

        # Draw rectangles around the detected people
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Convert frame to RGB (Tkinter uses RGB)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Convert the frame to Image for Tkinter
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the label widget with the new image
        label.imgtk = imgtk
        label.configure(image=imgtk)

    # Continuously update the camera feed
    label.after(10, show_frame)

# Start showing the frame
show_frame()

# Start the Tkinter event loop
root.mainloop()

# Release the camera when the GUI window is closed
cap.release()

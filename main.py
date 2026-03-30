import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, Button, Frame
from PIL import Image, ImageTk

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Load Known Image
known_image = cv2.imread("person1.jpg")
known_gray = cv2.cvtColor(known_image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(known_gray, 1.3, 5)
(x, y, w, h) = faces[0]
known_face = known_gray[y:y+h, x:x+w]

cap = None
running = False


def start_camera():
    global cap, running
    cap = cv2.VideoCapture(0)
    running = True
    status_label.config(text="Camera Running", fg="#00ff9c")
    show_frame()


def stop_camera():
    global running
    running = False
    status_label.config(text="Camera Stopped", fg="#ff4d4d")


def show_frame():
    global running

    if running:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (known_face.shape[1], known_face.shape[0]))

            diff = np.mean(cv2.absdiff(known_face, face))

            if diff < 70:
                name = "Nishant"
                color = (0,255,0)
            else:
                name = "Unknown"
                color = (0,0,255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)

        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        video_label.after(10, show_frame)


# GUI Window
root = tk.Tk()
root.title("Face Detection & Recognition")
root.geometry("900x650")
root.configure(bg="#0f172a")

# Title
title = Label(
    root,
    text="AI Face Detection & Recognition",
    font=("Segoe UI", 20, "bold"),
    bg="#0f172a",
    fg="#38bdf8"
)
title.pack(pady=15)

# Video Frame
video_frame = Frame(root, bg="#0f172a")
video_frame.pack()

video_label = Label(video_frame)
video_label.pack()

# Buttons Frame
btn_frame = Frame(root, bg="#0f172a")
btn_frame.pack(pady=20)

start_btn = Button(
    btn_frame,
    text="Start Camera",
    command=start_camera,
    font=("Segoe UI", 12, "bold"),
    bg="#22c55e",
    fg="white",
    width=15,
    bd=0,
    pady=8
)

start_btn.grid(row=0, column=0, padx=15)

stop_btn = Button(
    btn_frame,
    text="Stop Camera",
    command=stop_camera,
    font=("Segoe UI", 12, "bold"),
    bg="#ef4444",
    fg="white",
    width=15,
    bd=0,
    pady=8
)

stop_btn.grid(row=0, column=1, padx=15)

# Status Label
status_label = Label(
    root,
    text="Camera Stopped",
    font=("Segoe UI", 12),
    bg="#0f172a",
    fg="#ff4d4d"
)

status_label.pack()

# Footer
footer = Label(
    root,
    text="CodSoft AI Internship Project",
    font=("Segoe UI", 9),
    bg="#0f172a",
    fg="#64748b"
)

footer.pack(side="bottom", pady=10)

root.mainloop()
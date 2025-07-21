import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import threading
import random
import time
import numpy as np
from tensorflow.keras.models import load_model
import os

class ASLApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Sign Language to Text')
        self.root.geometry('900x700')
        self.root.resizable(False, False)

        # Webcam frame
        self.video_label = tk.Label(self.root, bg='gray', width=500, height=400)
        self.video_label.place(x=20, y=20)

        # Output text area
        self.text_area = tk.Text(self.root, font=('Arial', 18), width=40, height=5)
        self.text_area.place(x=20, y=440)

        # Buttons
        button_width = 180
        button_height = 50
        button_gap = 10

        self.clear_btn = tk.Button(self.root, text='Clear All', bg='#FFFACD', font=('Arial', 14), command=self.clear_text)
        self.clear_btn.place(x=20, y=600, width=button_width, height=button_height)
        self.save_btn = tk.Button(self.root, text='Save to a Text File', bg='#4CAF50', fg='white', font=('Arial', 14), command=self.save_text)
        self.save_btn.place(x=20 + button_width + button_gap, y=600, width=button_width + 40, height=button_height)
        self.quit_btn = tk.Button(self.root, text='Quit', bg='#F44336', fg='white', font=('Arial', 14), command=self.quit_app)
        self.quit_btn.place(x=20 + 2 * (button_width + button_gap) + 40, y=600, width=button_width, height=button_height)

        # Sidebar for ASL reference chart (placeholder)
        self.sidebar = tk.LabelFrame(self.root, text='Alphabet', font=('Arial', 12, 'bold'), width=300, height=630)
        self.sidebar.place(x=600, y=20)
        try:
            chart_img = Image.open("asl_chart.png")  # Update path if needed
            chart_img = chart_img.resize((260, 500), Image.ANTIALIAS)
            self.chart_photo = ImageTk.PhotoImage(chart_img)
            self.chart_label = tk.Label(self.sidebar, image=self.chart_photo)
            self.chart_label.image = self.chart_photo  # Prevent garbage collection
            self.chart_label.place(relx=0.5, rely=0.5, anchor='center')
        except Exception as e:
            self.chart_label = tk.Label(self.sidebar, text='[ASL Chart Not Found]', font=('Arial', 16))
            self.chart_label.place(relx=0.5, rely=0.5, anchor='center')

        # Webcam thread
        self.cap = None
        self.running = False
        self.last_frame = None  # Store the latest frame
        self.last_predicted = None
        self.last_append_time = 0
        self.start_webcam()

        # Predict button
        self.predict_btn = tk.Button(self.root, text='Predict', bg='#2196F3', fg='white', font=('Arial', 14), command=self.predict_gesture)
        self.predict_btn.place(x=200, y=380, width=120, height=40)

        self.model = None
        self.class_labels = []
        self.load_model_and_labels()

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.update_frame()

    def update_frame(self):
        if self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (500, 400))
                self.last_frame = frame.copy()
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk  # type: ignore[attr-defined]
                self.video_label.configure(image=imgtk)
            self.root.after(15, self.update_frame)  # Schedule next frame update

    def load_model_and_labels(self):
        model_path = os.path.join('models', 'asl_cnn.h5')
        if os.path.exists(model_path):
            self.model = load_model(model_path)
            # Get class labels from the dataset directory
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            datagen = ImageDataGenerator(rescale=1./255)
            gen = datagen.flow_from_directory('dataset', target_size=(64, 64), batch_size=1, class_mode='categorical')
            # Sort by class index
            self.class_labels = [None] * len(gen.class_indices)
            for label, idx in gen.class_indices.items():
                self.class_labels[idx] = label
        else:
            print('Trained model not found. Using mock predictions.')

    def predict_gesture(self):
        # Use a fixed ROI (top right, 300x300)
        if self.last_frame is not None:
            h, w, _ = self.last_frame.shape
            roi_size = 300
            margin = 20
            x_min = w - roi_size - margin
            y_min = margin
            x_max = w - margin
            y_max = margin + roi_size
            hand_roi = self.last_frame[y_min:y_max, x_min:x_max]
            if hand_roi.size > 0:
                # Show the ROI in a separate window for debugging
                cv2.imshow('ROI', hand_roi)
                cv2.waitKey(1)
                predicted = self.real_predict(hand_roi) if self.model else self.mock_predict(hand_roi)
                now = time.time()
                # Debounce: only append if different and 1s passed
                if predicted != self.last_predicted or (now - self.last_append_time) > 1.0:
                    if predicted == '<' or predicted.lower() == 'delete':  # Delete
                        current_text = self.text_area.get('1.0', tk.END)[:-2]
                        self.text_area.delete('1.0', tk.END)
                        self.text_area.insert(tk.END, current_text)
                    elif predicted == ' ' or predicted.lower() == 'space':
                        self.text_area.insert(tk.END, ' ')
                    elif predicted.lower() == 'nothing':
                        pass  # Do nothing for 'Nothing' class
                    else:
                        self.text_area.insert(tk.END, predicted)
                    self.last_predicted = predicted
                    self.last_append_time = now

    def real_predict(self, hand_roi):
        # Preprocess ROI for model
        img = cv2.resize(hand_roi, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        preds = self.model.predict(img)
        idx = np.argmax(preds)
        if self.class_labels:
            label = self.class_labels[idx]
            # Map special classes to symbols
            if label.lower() == 'space':
                return ' '
            if label.lower() == 'delete':
                return '<'
            return label
        return str(idx)

    def mock_predict(self, hand_roi):
        gesture_labels = [chr(i) for i in range(65, 91)] + [" ", "<"]  # A-Z, Space, Delete
        return random.choice(gesture_labels)

    def clear_text(self):
        self.text_area.delete('1.0', tk.END)

    def save_text(self):
        text = self.text_area.get('1.0', tk.END).strip()
        if not text:
            messagebox.showinfo('Info', 'No text to save!')
            return
        file_path = filedialog.asksaveasfilename(defaultextension='.txt', filetypes=[('Text Files', '*.txt')])
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
            messagebox.showinfo('Success', 'Text saved successfully!')

    def quit_app(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = ASLApp(root)
    root.mainloop()

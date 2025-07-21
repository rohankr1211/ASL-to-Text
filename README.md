# ASL to Text Translator

A modern, real-time American Sign Language (ASL) to Text Translator built with Python, OpenCV, MediaPipe, TensorFlow/Keras, and Tkinter.
This application uses your webcam to recognize ASL hand signs (A–Z, Space, Delete, Nothing) and translates them into text, providing a user-friendly desktop interface and an integrated ASL reference chart.

---

## 🚀 Features

- Real-time hand detection and gesture recognition using webcam
- Deep learning model (CNN) trained on ASL alphabet dataset
- Interactive Tkinter GUI with live video feed and output text area
- Predicts A–Z, Space, Delete, and Nothing gestures
- “Clear All”, “Save to a Text File”, and “Quit” buttons
- Sidebar with ASL reference chart for easy lookup
- Debounce logic to prevent repeated letters
- (Optional) Text-to-speech and other enhancements

---

## 🛠️ Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rohankr1211/ASL-to-Text.git
   cd ASL-to-Text
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the ASL Alphabet Dataset:**
   - [Kaggle: ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
   - Place the dataset in the `dataset/` folder and use the provided scripts in `utils/` to organize and clean it.

4. **Train the model:**
   ```bash
   python utils/train_model.py
   ```
   - The trained model will be saved in `models/asl_cnn.h5`.

5. **Run the application:**
   ```bash
   python gui.py
   ```

---

## 🖥️ Usage Guide

- Show your right hand in the orange ROI box on the webcam feed.
- Refer to the ASL chart in the sidebar for correct hand signs.
- Press the “Predict” button to recognize the current sign and append it to the text area.
- Use “Clear All” to reset the text area.
- Use “Save to a Text File” to save your translated text.
- Use “Quit” to exit the application.

---

## 📸 Screenshots

> *(Add your screenshots in the `screenshots/` folder and embed them below)*

![GUI Screenshot](screenshots/gui_output.png)
![ASL Chart Sidebar](screenshots/asl_chart_sidebar.png)

---

## 📚 Credits

- **ASL Alphabet Dataset:** [Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Libraries:** OpenCV, MediaPipe, TensorFlow, Keras, Pillow, Tkinter

---

## 📄 License

This project is licensed under the MIT License.

# ✋**Sign Language Translator (Real-Time Hand Gesture Recognition)**

This project is a **real-time American Sign Language (ASL) letter recognition system** built using **OpenCV**, **cvzone**, and a **CNN-based classifier**.
It captures hand gestures from a webcam, processes them, classifies them into letters, and displays the predicted text live.

This work is inspired by the struggle faced by sign-language users to communicate with non-signers, motivating a system that identifies **individual ASL letters** and displays them instantly.

---

## 📌 **Features**

* Real-time **hand detection** using cvzone’s `HandDetector`
* Automatic **image preprocessing** and **hand cropping** 
* CNN-based gesture classification (`keras_model.h5`)
* **Consistency check** so only held gestures (3 seconds) are accepted as valid characters 
* Live **predicted-text overlay**
* FPS tracking and performance metrics
* Easy to extend to full-word and sentence recognition

---

## 🧠 **How the System Works**

### **1️⃣ Input Capture (Image Acquisition)**

A webcam captures video frames in real time.
Using `HandDetector`, the system finds a bounding box around the hand and extracts the region of interest.

### **2️⃣ Preprocessing & Hand Isolation**

Each frame undergoes:

* Cropping
* Centering on a 300×300 white canvas
* Resizing while maintaining aspect ratio
* Background reduction

This ensures consistent input for the classifier.
The full preprocessing logic is implemented in `dataCollection.py` and `testing.py`

### **3️⃣ Classification**

A CNN model (`keras_model.h5`) trained on ASL letters predicts the gesture.
Labels include: **H, E, L, O, A, 5, 6**.

### **4️⃣ Consistency & Text Prediction**

To avoid flickering, the system only accepts a gesture if the user holds it for **3 seconds**.
Then it appends the recognized letter to `recognized_text`.

### **5️⃣ Output**

The top bar displays the final recognized text, while the main screen shows:

* Hand bounding box
* Predicted label
* FPS

---

## 💻 **Tech Stack**

| Component         | Technology         |
| ----------------- | ------------------ |
| Programming       | Python             |
| Vision Processing | OpenCV             |
| Hand Detection    | cvzone + MediaPipe |
| ML Model          | TensorFlow / Keras |
| Hardware          | Standard webcam    |

---

## 📁 **Project Structure**

```
Sign_Language_Translator/
│── Data/                 # Dataset used for training
│── Model/                # keras_model.h5 + labels.txt
│── Reports/              # Original technical report
│── dataCollection.py     # Script for capturing gesture images
│── testing.py            # Real-time gesture prediction
│── requirements.txt      # Dependencies
│── README.md
```

---

## 🧪 **Results & Performance**

### ✔ **Recognition Accuracy**

* High accuracy for clear, distinct gestures
* Slight errors on visually similar letters
* Works best under **consistent lighting**

### ✔ **Speed**

* Provides near-instant predictions
* FPS displayed live on screen
* Low latency gesture-to-text translation

### ✔ **Environmental Testing**

* Best performance: Simple backgrounds + good lighting
* Reduced accuracy: Dim light or cluttered backgrounds

---

## 🧪 **To Collect Training Data**

Use:

```
python dataCollection.py
```

Press **S** to save images.

---

## ▶️ **To Run the Translator**

```
python testing.py
```

Press **ESC** to exit.

---

## 📦 **Installation**

### 1️⃣ Create environment (optional)

```
python -m venv venv
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

This installs:

* opencv-python
* cvzone
* numpy
* tensorflow

---

## 📈 **System Flow**

1. Start
2. Capture image frame
3. Preprocessing (lighting, resizing, background subtraction)
4. Feature extraction (contours + shape features)
5. CNN prediction
6. Display predicted letter
7. Loop

This is exactly what `testing.py` implements.

---

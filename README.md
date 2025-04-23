# 👁️‍🗨️ Facial Bias Detection in Healthcare Using AI

A deep dive into gender bias in facial recognition systems, specifically tailored to healthcare applications.  
This project demonstrates how to build, test, and reduce bias in AI gender classification models while preserving privacy using differential noise techniques.

---

## 🧠 Key Features

- 🧍 Gender classification using CNN with OpenCV & Caffe
- 🔒 Privacy-preserving noise addition (Salt & Pepper noise)
- 📊 Bias detection with `classification_report` and `heatmaps`
- 📁 Dataset from UTKFace (age, gender, race-labeled facial images)
- 💡 Report shows how noise reduces bias & accuracy proportionally

---

## 🧰 Tech Stack

- Python, OpenCV, scikit-learn, seaborn, matplotlib, NumPy
- Caffe model (`deploy_gender.prototxt`, `gender_net.caffemodel`)
- Differential Privacy technique: Salt & Pepper Noise
- UTKFace Dataset (10,000+ labeled face images)

---

## 📁 Project Structure


---

## 🧪 How to Run

1. Install required libraries:
   ```bash
   pip install -r requirements.txt
    ```

2. Place UTKFace images inside data/images/
(Structure: data/images/*.jpg with filenames like 25_0_2_20170116174525125.jpg.chip.jpg)

3. Run the main file:
```
python bias_ai.py
```
4. View the classification report and bias heatmap in the output.

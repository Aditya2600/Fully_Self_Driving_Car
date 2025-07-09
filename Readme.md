# 🚗 Self-Driving Car Project

This is an end-to-end self-driving car pipeline that combines:

- 🧠 **CNN Steering Angle Prediction** using TensorFlow 1.x
- 🛣️ **Lane + Object Segmentation** using YOLOv8 (Ultralytics)
- 📹 **Real-time Visualization** using OpenCV
- 🔁 **Threaded Inference** for performance

---

## 🎬 Demo

![Demo](demo_output.gif)

---

## 🛠️ Installation

```bash
git clone https://github.com/Aditya2600/Self_Driving_Car.git
cd Self_Driving_Car
pip install -r requirements.txt


⸻

🚀 Run Inference Simulation

python src/inference/run_fsd_inference.py

Make sure the following models and files exist:

saved_models/
├── regression_model/model.ckpt
├── lane_segmentation_model/best.pt
├── object_detection_model/best.pt

data/
├── driving_dataset/0.jpg, 1.jpg, ...
├── steer-wheel.png


⸻

🧠 Training the Steering Angle Model

Training is based on NVIDIA’s self-driving architecture.

python model_training/train_steering_angle/train.py

Logs are saved for TensorBoard:

tensorboard --logdir=model_training/train_steering_angle/logs


⸻

🗂️ Project Structure

src/
├── inference/
│   └── run_fsd_inference.py
├── models/
│   └── model.py  # CNN architecture
saved_models/
data/


⸻

📦 Dependencies

See requirements.txt, includes:
	•	tensorflow==1.15
	•	ultralytics
	•	opencv-python
	•	numpy

⸻

👤 Author

Aditya Meshram

⸻

📄 License

This project is licensed under the MIT License.

---



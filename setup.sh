#!/bin/bash

echo "Downloading saved models..."

mkdir -p saved_models/lane_segmentation_model
wget -O saved_models/lane_segmentation_model/best_yolo11_lane_segmentation.pt "https://drive.google.com/file/d/1oRB1lRfrE9xLkG089mOYAMgnodIZsFWk/view?usp=sharing"

mkdir -p saved_models/object_detection_model
wget -O saved_models/object_detection_model/yolo11m-seg.pt "https://drive.google.com/file/d/1oSvavrDpkzI8OH6F2XQfZxfM8Bf5uX6D/view?usp=sharing"

mkdir -p saved_models/regression_model
echo "Steering model not bundled. Train with model_training/train_steering_angle/train.py to create saved_models/regression_model/model.pth."

echo "✅ All models downloaded."

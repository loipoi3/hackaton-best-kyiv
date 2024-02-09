from ultralytics import YOLO
import torch
import os

test_data_path = os.getenv('TEST_DATA_PATH')

model = YOLO('../models/yolov8n.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model.to(device)
results = model(test_data_path)
print(results)

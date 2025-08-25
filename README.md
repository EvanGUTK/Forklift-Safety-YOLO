# Project Mira – Forklift Safety (YOLO-Based)

![Project Mira Placeholder](path/to/image.png)

---

## 📌 Overview
**Project Mira** is a **YOLO (You Only Look Once)–based AI program** designed to enhance forklift safety in industrial environments. By combining advanced computer vision with real-time sensor integration, this system helps **prevent forklift-related injuries** by detecting humans, obstacles, and foreign objects in the vehicle’s path.  

The program runs on **CUDA-capable devices** such as the **NVIDIA Jetson Xavier**, leveraging its GPU and GPIO ports to directly communicate with forklift electronics (e.g., governors, sirens, and braking systems).  

Inspired by **Tesla’s Full Self-Driving (FSD)** principles, Project Mira provides **real-time perception and autonomous safety intervention** using a combination of **CSI cameras, ultrasonic sensors, mappers, and LiDAR**.  

---

## 🚀 Features
- **Camera Integration**: Supports 1 to 4 cameras for creating a **dynamic bird’s-eye view**.  
- **Ultrasonic Sensors**: Uses 1 to 4 sensors for short-range **object detection and mapping**.  
- **3D Mapping**: Employs a mapper to build a **3D HD layout** of the forklift’s environment.  
- **LiDAR Support**: Optional integration for enhanced accuracy and detection range.  
- **CUDA Acceleration**: Runs on **NVIDIA Jetson Xavier** with CUDA toolkit support.  
- **Active Forklift Control**: Direct GPIO integration allows the system to signal:  
  - Speed governor  
  - Safety siren  
  - Warning lights  
  - Full forklift stop until the obstruction is cleared  

---

## 🧩 Technical Packages

### Simple CV
```python
import cv2
from ultralytics import YOLO
```
### Advance CV
```python
import os
import time
import math
import uuid
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import List, Optional, Union, Tuple
import numpy as np
import torch
import torch.cuda as cuda
```

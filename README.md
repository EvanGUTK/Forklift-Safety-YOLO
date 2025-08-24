# Project Mira

## Forklift Safety (YOLO - Based)
- This program uses a library called YOLO (You Only Look Once) This library is compiled of images to identify objects and person to help prevent forklift injuries. 
- The program can be ran using a CUDA device such, the NVIDIA Jetson along with its GPIO ports. 
- The GPIO ports are used to speak with said forklift to engage its governor  and electronics. 
- The Jetson will utilize a mapper, ultrasonic sensors, CSI cameras, and LiDAR for mapping is surrounding in memory and live detection.
## Overview
- This project was inspired from Tesla's FSD. This program is able to detect humans, obstacles, and foreign objects. Once detected the Jetson can then signal the speed governor, siren, lights, or even fully stop the forklift until object or person is out of the way. 

## Features
- Uses 1 to 4 cameras in unison to create a birds-eye-view
- Uses 1 to 4 ultrasonic sensors to map out its surroundings
- Uses a single mapper to create a 3D HD layout of forklifts surroundings
- Possible LiDAR for better detection
- Jetson Xavier for CUDA toolkit

## Packages (Simple CV)
- import cv2
- from ultralytics import YOLO

## Packages (Indepth CV)
- import os
- import time
- import math
- import uuid
- from pathlib import Path
- from datetime import datetime
- from collections import deque
- from typing import List, Optional, Union, Tuple
- import numpy as np
- import torch
- import torch.cuda as cuda

## How to
- Run code using your favorite IDE to run Python code
- e.g. Visual Studio Code, Visual Studio, PyCharm, Spyder, ect
- Download .py to said IDE and ensure proper packages are installed for which version you choose. If you have a CUDU supported device ensure you install CUDA package that matches your IDE .py firmware. If you are using CPU ensure torchvision is installed.

## Features

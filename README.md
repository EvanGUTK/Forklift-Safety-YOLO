# Project Mira

## Forklift Safety (YOLO - Based)
- This program uses a library called YOLO (You Only Look Once) This library is compiled of images to identify objects and person to help prevent forklift injuries. 

- The program can be ran using a CUDA device such, the NVIDIA Jetson along with its GPIO ports. 

- The GPIO ports are used to speak with said forklift to engage its governor  and electronics. 

- The Jetson will utilize a mapper, ultrasonic sensors, CSI cameras, and LiDAR for mapping is surrounding in memory and live detection.

## Overview
- This project was inspired from Tesla's FSD. This program is able to detect humans, obstacles, and foreign objects. Once detected the Jetson can then signal the speed governor, siren, lights, or even fully stop the forklift until object or person is out of the way. 

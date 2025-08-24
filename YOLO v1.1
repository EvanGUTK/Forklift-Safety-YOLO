import os
import cv2
import torch
from ultralytics import YOLO

def pick_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def main():
    weights = os.getenv("YOLO_WEIGHTS", "yolov8s.pt")
    conf    = float(os.getenv("CONF", "0.35"))
    iou     = float(os.getenv("IOU",  "0.60"))
    imgsz   = int(os.getenv("IMG",   "960"))
    device  = pick_device()

    model = YOLO(weights)
    results = model.predict(
        source=0,
        stream=True,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        classes=[0],
        device=device,
        verbose=False
    )

    cv2.namedWindow("YOLOv8 - (v1.1)", cv2.WINDOW_NORMAL)

    try:
        for r in results:
            frame = r.plot()
            cv2.imshow("YOLOv8 - (v1.1)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

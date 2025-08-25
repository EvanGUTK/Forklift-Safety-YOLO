# Proper libraries to import to ensure code runs
# When you initially run code you will be given option to download libraries
import cv2
from ultralytics import YOLO

def main():

    model = YOLO("yolov8n.pt")

    results = model.predict(source=0, stream=True, imgsz=640, conf=0.25, classes=[0])

    cv2.namedWindow("YOLOv8", cv2.WINDOW_NORMAL)

    try:
        for r in results:
            frame = r.plot()
            cv2.imshow("YOLOv8", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

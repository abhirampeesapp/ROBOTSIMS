from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8n.pt")  # fast model

INPUT_FOLDER = "captured_frames"
OUTPUT_FOLDER = "output_frames"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def run_detection():
    for file in os.listdir(INPUT_FOLDER):
        if file.endswith(".jpg") or file.endswith(".png"):
            path = os.path.join(INPUT_FOLDER, file)

            frame = cv2.imread(path)
            results = model(frame)

            annotated = results[0].plot()

            out_path = os.path.join(OUTPUT_FOLDER, file)
            cv2.imwrite(out_path, annotated)

            print(f"Processed: {file}")

if __name__ == "__main__":
    run_detection()
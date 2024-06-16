import cv2
from ultralytics import YOLO

def main():
    cap = cv2.VideoCapture(0)

    model_path = "yolov8 project/yolov8/yolov8/person_detection/yolov8n.pt"
    model = YOLO(model_path)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame)

        annotated_frame = frame.copy()
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("YOLOv8", annotated_frame)

        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

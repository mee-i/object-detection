from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")
results = model(source="cars.webm", stream=True)

for result in results:
    frame = result.plot()
    cv2.imshow("YOLO Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

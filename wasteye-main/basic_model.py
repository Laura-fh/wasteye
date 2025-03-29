from ultralytics import YOLO
#.
# Load a COCO-pretrained YOLOv8n model
#model = YOLO("yolov8n.pt")

model = YOLO("yolov8n-cls.pt")

    # Display model information (optional)
model.info()

    # Train the model on the COCO8 example dataset for 100 epochs
#results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

 # Define remote image or video URL
source = "https://ultralytics.com/images/bus.jpg"

    # Run inference on the source
results = model(source)  # list of Results objects
probs=results[0].probs
print(probs)
'''
# Load a COCO-pretrained YOLOv8n model and train it on the COCO8 example dataset for 100 epochs
yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

    # Load a COCO-pretrained YOLOv8n model and run inference on the 'bus.jpg' image
yolo predict model=yolov8n.pt source=path/to/bus.jpg


How do I train a YOLOv8 model?
=== "Python"

    ```python
    from ultralytics import YOLO

    # Load a COCO-pretrained YOLOv8n model
    model = YOLO("yolov8n.pt")

    # Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
    ```

=== "CLI"

    ```bash
    yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640
    ```

'''

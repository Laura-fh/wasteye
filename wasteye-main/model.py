from ultralytics import YOLO
from PIL import Image
import os
import shutil


def train_model(model=YOLO("yolov10n.pt"), data_yaml="wasteye-main/yolo_config.yaml", epochs=10, img_size=416):
    '''
    Train a YOLOv10 model on our custom dataset.
    Args:
        model (<class 'ultralytics.yolo.engine.model.YOLO'>): YOLOv10 model.
        data_yaml (str): Path to the dataset YAML file.
        epochs (int): Number of training epochs.
        img_size (int): Image size for training. Default is 416.
    Returns:
        results (<class 'ultralytics.engine.results.Results'>):
        Contains training details (metrics, loss values, etc).
    '''
    # Load a pretrained object detection model
    print(f"\n🚀 Training model...\n")
    results = model.train(data=data_yaml, epochs=epochs, imgsz=img_size)
    print("✅ Training Complete!")
    return results

def move_weights():
    '''
    Move the best.pt file to a new folder called weights_path.
    '''
    # Create a new folder for the weights
    os.makedirs("wasteye-main/weights_path", exist_ok=True)

    # Move the best.pt file to the new folder
    shutil.move("wasteye-main/runs/detect/train/weights/best.pt", "wasteye-main/weights_path/best.pt")

    # Check if the file was moved successfully
    if os.path.exists("wasteye-main/weights_path/best.pt"):
        print("✅ best.pt moved to weights_path!")
        shutil.rmtree("wasteye-main/runs")
        print("✅ Original weights folder deleted!")
    else:
        print("❌ Failed to move best.pt!")



def load_model(weights_path="wasteye-main/weights_path/best.pt"):
    '''
    Load the trained YOLOv10 model's weights.''
    Args:
        weights_path (str): Path to the trained model weights.
    Returns:
        model (<class 'ultralytics.yolo.engine.model.YOLO'>): Loaded YOLOv10 model.
    '''
    # Load the model with the specified weights
    print(f"\n🚀 Loading model...\n")
    model = YOLO(weights_path)
    print("✅ Model Loaded!")
    return model


def evaluate_model(model):
    '''
    Evaluate the trained YOLOv10 model on the validation set.
    Args:
        model (<class 'ultralytics.yolo.engine.model.YOLO'>): Loaded YOLOv10 model.
    Returns:
        metrics (<class 'ultralytics.engine.results.Results'>): Evaluation metrics.
    '''
    model = load_model()
    print(f"\n🚀 Evaluating model...\n")
    metrics = model.val(imgsz=416, conf=0.15, iou=0.60)
    print("✅ Evaluation Complete!")
    return metrics



def pred(model, images_path="~/gcs/images/test", conf=0.20, imgsz=416, save=True):
    '''
    Make predictions using the trained YOLOv10 model.
    Args:
        model (<class 'ultralytics.yolo.engine.model.YOLO'>): Loaded YOLOv10 model weights.
        images_path (str): Local mounted path for GCS images for prediction.
        conf (float): Confidence threshold for predictions. Default is 0.20.
        imgsz (int): Image size for prediction. Default is 416.
        save (bool): Whether to save the predictions. Default is True.
    Returns:
        pred (<class 'ultralytics.engine.results.Results'>): Prediction results (an array for each image).
    '''
    model = load_model()
    print(f"\n🚀 Making predictions...\n")
    pred = model.predict(source=images_path, conf=conf, imgsz=imgsz, save=save)
    print("\n✅ prediction done: \n")
    if save:
        # The predictions will be saved in the 'runs/detect/predict' folder
        saved_results_folder = "runs/detect/predict/"
        print(f"\n✅ Predictions saved in: {saved_results_folder}")
        return pred
    else:
        return pred


if __name__ == '__main__':
    train_model()
    move_weights()
    evaluate_model()
    #pred()

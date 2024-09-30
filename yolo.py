from ultralytics import YOLO

def train_yolo(pseudo_label_folder, unlabelled_image_folder):
    model = YOLO("yolov8n.pt")
    model.train(
        data={
            'train': unlabelled_image_folder,
            'val': unlabelled_image_folder
        },
        epochs=50,
        imgsz=640,
        batch_size=16
    )
    
    model.save('yolo_model.pth')

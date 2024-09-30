import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from utils import coco_dataset, save_pseudo_labels
from PIL import Image
import os

def get_faster_rcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train_faster_rcnn(dataset_name, num_epochs=10):
    if dataset_name == 'coco_person':
        dataset, num_classes = coco_dataset()
    
    # Load dataset and create a data loader
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
    
    # Get Faster R-CNN model
    model = get_faster_rcnn_model(num_classes)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    
    # Define optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        
        scheduler.step()
        print(f'Epoch {epoch} completed.')
    
    # Save the trained model
    torch.save(model.state_dict(), 'fasterrcnn_model.pth')

def generate_pseudo_labels(unlabelled_image_folder, output_folder):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_faster_rcnn_model(num_classes=2)  # Change num_classes as per dataset
    model.load_state_dict(torch.load('fasterrcnn_model.pth'))
    model.eval().to(device)

    os.makedirs(output_folder, exist_ok=True)

    for img_file in os.listdir(unlabelled_image_folder):
        img_path = os.path.join(unlabelled_image_folder, img_file)
        image = Image.open(img_path)
        image_tensor = F.to_tensor(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)[0]
        
        save_pseudo_labels(outputs, img_file, output_folder)

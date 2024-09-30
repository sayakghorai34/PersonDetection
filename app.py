from flask import Flask, request, jsonify
from threading import Thread
from faster_rcnn import train_faster_rcnn, generate_pseudo_labels
from yolo import train_yolo

app = Flask(__name__)

@app.route('/train-faster-rcnn', methods=['POST'])
def train_faster_rcnn_api():
    dataset = request.json['dataset']  # e.g., 'coco_person'
    
    def train_in_background():
        train_faster_rcnn(dataset)

    thread = Thread(target=train_in_background)
    thread.start()
    return jsonify({"status": "Faster R-CNN training started"}), 200

@app.route('/pseudo-label', methods=['POST'])
def pseudo_label_api():
    unlabelled_image_folder = request.json['unlabelled_image_folder']
    output_folder = request.json['output_folder']
    
    def generate_pseudo_labels_in_background():
        generate_pseudo_labels(unlabelled_image_folder, output_folder)

    thread = Thread(target=generate_pseudo_labels_in_background)
    thread.start()
    return jsonify({"status": "Pseudo-label generation started"}), 200

@app.route('/train-yolo', methods=['POST'])
def train_yolo_api():
    pseudo_label_folder = request.json['pseudo_label_folder']
    unlabelled_image_folder = request.json['unlabelled_image_folder']
    
    def train_yolo_in_background():
        train_yolo(pseudo_label_folder, unlabelled_image_folder)

    thread = Thread(target=train_yolo_in_background)
    thread.start()
    return jsonify({"status": "YOLO training started"}), 200

if __name__ == '__main__':
    app.run(debug=True)

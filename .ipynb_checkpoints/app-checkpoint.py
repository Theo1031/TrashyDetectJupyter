from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = Flask(__name__)

def load_model(model_path):
    # Initialize the model architecture to match the training setup
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Adjust for binary classification (cats vs. dogs)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

model_path = 'model_best.pth'  # Adjust the path as necessary
model = load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Convert the Image from bytes to a PIL Image
        image_bytes = request.files['image'].read()
        image = Image.open(io.BytesIO(image_bytes))

        # Apply the same transformations as used during model training
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            outputs = model(image)
            _, preds = torch.max(outputs, 1)

        # Define labels
        labels = ['Cat', 'Dog']  # Assuming index 0 is Cat and index 1 is Dog
        prediction = labels[preds.item()]

        return jsonify({'prediction': prediction, 'confidence': torch.softmax(outputs, dim=1).max().item()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
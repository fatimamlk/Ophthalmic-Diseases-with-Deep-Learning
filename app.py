from flask import Flask, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename
import os
import pickle
from PIL import Image
import torch
import torchvision.transforms as transforms
from Model import CNN

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Load the machine learning model from the pickle file
model_path = 'model.pkl'
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image

def predict_image(image_path, model):
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(image_tensor)
    probability = output.item()  # Get the probability value
    result = "There is no retinopathy diabetics" if probability >= 0.5 else "There is  retinopathy diabetics"  # Apply threshold to get the final prediction
    return result, probability

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part in the request"
    file = request.files['file']
    if file.filename == '':
        return "No file selected for uploading"
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Make predictions
        result, probability = predict_image(file_path, model)
        
        # Return prediction result and file upload success message
        return f"File uploaded successfully: <a href='{url_for('uploaded_file', filename=filename)}'>View File</a><br>Prediction: {result}, Probability: {probability}"
    else:
        return "Allowed file types are png, jpg, jpeg, gif"

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)

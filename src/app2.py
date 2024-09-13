import os
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from flask_mail import Mail, Message

app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Configuration for Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # Use Gmail's SMTP server
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your_email@gmail.com'  # Replace with your email
app.config['MAIL_PASSWORD'] = 'your_password'  # Replace with your email password
app.config['MAIL_DEFAULT_SENDER'] = 'your_email@gmail.com'  # Default sender

mail = Mail(app)

# Initialize the model (using BLIP as an example)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Check if the uploaded file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Main route for image upload and email submission
@app.route('/')
def upload_form():
    return render_template('upload_email.html')

# Route to handle image uploads and sending email
@app.route('/', methods=['POST'])
def upload_and_send():
    if 'file' not in request.files or 'email' not in request.form:
        return 'No file part or email part'
    
    file = request.files['file']
    email = request.form['email']

    if file.filename == '':
        return 'No selected file'
    if not email:
        return 'No email provided'
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Analyze the image using the model
        image = Image.open(filepath).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        # Send the result via email
        msg = Message('Your Image Analysis Result', recipients=[email])
        msg.body = f'Uploaded Image: {filename}\nGenerated Caption: {caption}'
        with app.open_resource(filepath) as fp:
            msg.attach(filename, "image/jpeg", fp.read())
        mail.send(msg)

        return f'Analysis complete. Results sent to {email}.'

    return 'File not allowed'

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)

import os
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from flask_mail import Mail, Message
from PIL import Image
from main import run_example

app = Flask(__name__)

# Directory to save uploaded files
UPLOAD_FOLDER = 'src/static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions for uploaded files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Configuration for Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # Use Gmail's SMTP server
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your_email@gmail.com'  # Replace with your email
app.config['MAIL_PASSWORD'] = 'your_password'  # Replace with your email password
app.config['MAIL_DEFAULT_SENDER'] = 'your_email@gmail.com'  # Default sender

mail = Mail(app)

# Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Main route for upload form
@app.route('/')
def upload_form():
    return render_template('upload_folder.html')

# Route to handle folder uploads
@app.route('/', methods=['POST'])
def upload_folder():
    if 'files[]' not in request.files:
        return 'No files part'
    
    files = request.files.getlist('files[]')
    uploaded_filenames = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image = Image.open(filepath).convert("RGB")
            task_prompt = '<OPEN_VOCABULARY_DETECTION>'
            results = run_example(image, task_prompt, text_input="a green car")
            print(results)
            uploaded_filenames.append(filename)
    
    if not uploaded_filenames:
        return 'No files uploaded or invalid file types'
    
    return f"Uploaded files: {', '.join(uploaded_filenames)}"

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)

import os
import subprocess
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Directory to save uploaded files
UPLOAD_FOLDER = 'src/static/uploads/'
LOG_FOLDER = 'logs/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['LOG_FOLDER'] = LOG_FOLDER

# Allowed extensions for uploaded files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Main route for file upload
@app.route('/')
def upload_form():
    return render_template('upload_message.html')

# Route to handle image uploads and email
@app.route('/', methods=['POST'])
def upload_file():
    # Capture email from the form
    email = request.form.get('email')
    what_to_detect = request.form.get('what_to_detect')
    
    if not email:
        return 'Email is required'
    if not what_to_detect:
        return 'Please specify what to detect in the image.'
    # Handle file upload
    if 'files[]' not in request.files:
        return 'No files part'
    
    files = request.files.getlist('files[]')
    if len(files) == 0:
        return 'No selected files'
    # Create a unique folder to store this user's uploaded images
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], email)
    os.makedirs(user_folder, exist_ok=True)
    
    # Save each file
    filepaths = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(user_folder, filename)
            file.save(filepath)
            filepaths.append(filepath)

    if not filepaths:
        return 'No valid files uploaded.'

    # Display message to the user
    message = f"We will send the pictures to {email} in a few hours."

    # Call main.py asynchronously, passing the uploaded file path and email
    run_background_process(user_folder, email, what_to_detect)

    return render_template('message.html', message=message)

# Function to run main.py in the background and capture logs
def run_background_process(filepath, email, what_to_detect):
    # Ensure logs directory exists
    os.makedirs(app.config['LOG_FOLDER'], exist_ok=True)

    # Log file path
    log_file = os.path.join(app.config['LOG_FOLDER'], 'processing.log')

    # Open the log file in append mode
    with open(log_file, 'a') as f:
        # Run main.py in the background, pass filepath and email, and redirect stdout/stderr to the log file
        subprocess.Popen(['python', 'src/main.py', filepath, what_to_detect], stdout=f, stderr=subprocess.STDOUT)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)

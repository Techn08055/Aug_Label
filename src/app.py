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

# Route to handle image uploads
@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Display message to the user
        message = "We will send you pictures in a few hours."

        # Call main.py asynchronously, passing the uploaded file path
        run_background_process(filepath)

        return render_template('message.html', message=message)

    return 'File not allowed'

# Function to run main.py in the background
def run_background_process(filepath):
    # Run main.py in the background and pass the filepath as an argument
    os.makedirs(app.config['LOG_FOLDER'], exist_ok=True)

    # Log file path
    log_file = os.path.join(app.config['LOG_FOLDER'], 'processing.log')

    # Open the log file in append mode
    with open(log_file, 'a') as f:
        # Run main.py in the background, pass filepath, and redirect stdout/stderr to the log file
        subprocess.Popen(['python', 'src/main.py', filepath], stdout=f, stderr=subprocess.STDOUT)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)

from app import app
# app.py
from flask import Flask, request, jsonify, render_template, redirect, g, url_for, send_from_directory
import sqlite3
import os
import glob
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw
import random

DATABASE = 'responses.db'

UPLOAD_FOLDER = 'uploaded_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


GENERATED_FOLDER = 'generated_images'  # Folder for generated images
os.makedirs(GENERATED_FOLDER, exist_ok=True)
GENERATED_FOLDER = os.path.join(os.getcwd(), 'generated_images')
app.config['GENERATED_FOLDER'] = GENERATED_FOLDER

# Database connection
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# Initialize the database
def init_db():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        
        # Create the user_info table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_info (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
            )
        ''')

        # Create the suspect_descriptions table for structured survey responses
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS suspect_descriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL UNIQUE,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,

                -- General Appearance
                gender TEXT,
                age TEXT, 
                body_build TEXT, 
                height TEXT,
                distinctive_features TEXT,

                -- Facial Features
                face_shape TEXT,
                hair_color TEXT, 
                hair_style TEXT,
                forehead TEXT,
                eyebrows TEXT,
                eyes TEXT,
                nose TEXT,
                mouth TEXT,
                teeth TEXT,
                chin_jawline TEXT,
                facial_hair TEXT,

                -- Skin and Complexion
                skin_tone TEXT,
                skin_blemishes TEXT,
                wrinkles TEXT,

                -- Clothing and Accessories
                clothing TEXT,
                accessories TEXT,
                glasses_description TEXT,

                -- Additional Details
                standout_features TEXT,
                confidence_level TEXT,

                summary TEXT, 
                uploaded_image TEXT,
                generated_images TEXT,
                       
                FOREIGN KEY(user_id) REFERENCES user_info(user_id)
            )
        ''')
        db.commit()

# Route to display the welcome page
@app.route('/')
def home():
    return render_template('home.html')

# Route to redirect from welcome page to personal info page
@app.route('/start', methods=['POST'])
def start():
    return redirect(url_for('personal_info'))

# Route for personal information form
@app.route('/personal-info', methods=['GET', 'POST'])
def personal_info():
    if request.method == 'POST':
        data = request.get_json()
        name = data['name']

        db = get_db()
        cursor = db.cursor()
        cursor.execute(
            'INSERT INTO user_info (name) VALUES (?)',
            (name,)
        )
        db.commit()

        # Return user_id to the client
        user_id = cursor.lastrowid
        return jsonify({'user_id': user_id, 'message': 'Personal info saved successfully'}), 200

    return render_template('personal_info.html')

# Survey page
@app.route('/survey')
def survey():
    return render_template('survey.html')

# Route to save survey responses
@app.route('/submit', methods=['POST'])
def submit_response():
    data = request.get_json()
    user_id = data['userId']
    responses = data['responses']

    db = get_db()
    cursor = db.cursor()

    for category, response in responses.items():
        if response:  # Only update if response is provided
            cursor.execute(f'''
                INSERT INTO suspect_descriptions (user_id, {category})
                VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET {category} = excluded.{category}
            ''', (user_id, response))

    db.commit()
    generate_description(user_id)
    return jsonify({'message': 'Responses saved successfully'}), 200


def is_survey_complete(user_id):
    """Check if all necessary responses for the survey have been completed by the user."""
    cursor = get_db().cursor()
    cursor.execute('''
        SELECT gender, age, body_build, height, distinctive_features, face_shape, hair_color, hair_style, eyes, nose, 
                   mouth, teeth, chine_jawline, facial_hair, skkine_tone, skin_blemishes, wrinkles, clothing, accesories, glasses_description
                   standout_features, confidence_level
        FROM suspect_descriptions WHERE user_id = ?
    ''', (user_id,))
    result = cursor.fetchone()
    return all(result)  # True if all necessary fields are filled

def generate_description(user_id):
    """Generate a description paragraph based on survey responses for the given user and store it in the database."""
    cursor = get_db().cursor()
    
    # Retrieve survey responses
    cursor.execute('''
        SELECT gender, age, body_build, height, distinctive_features, face_shape, hair_color, hair_style, forehead, 
               eyebrows, eyes, nose, mouth, teeth, chin_jawline, facial_hair, skin_tone, skin_blemishes, wrinkles, 
               clothing, accessories, glasses_description, standout_features, confidence_level
        FROM suspect_descriptions WHERE user_id = ?
    ''', (user_id,))
    row = cursor.fetchone()

    if not row:
        return "No description available for this user."

    # Generate the description
    parts = [
        f"The suspect is described as {row['gender']}" if row['gender'] else "",
        f"around {row['age']} years old" if row['age'] else "",
        f"with a {row['body_build']} build" if row['body_build'] else "",
        f"approximately {row['height']} tall" if row['height'] else "",
        f"with distinctive features like {row['distinctive_features']}" if row['distinctive_features'] else "",
        f"The face shape is {row['face_shape']}" if row['face_shape'] else "",
        f"hair color is {row['hair_color']}" if row['hair_color'] else "",
        f"hair style is {row['hair_style']}" if row['hair_style'] else "",
        f"with a {row['forehead']} forehead" if row['forehead'] else "",
        f"and {row['eyebrows']} eyebrows" if row['eyebrows'] else "",
        f"The eyes are {row['eyes']}" if row['eyes'] else "",
        f"with a {row['nose']} nose" if row['nose'] else "",
        f"a {row['mouth']} mouth" if row['mouth'] else "",
        f"and {row['teeth']} teeth" if row['teeth'] else "",
        f"The jawline is {row['chin_jawline']}" if row['chin_jawline'] else "",
        f"with {row['facial_hair']} facial hair" if row['facial_hair'] else "",
        f"The skin tone is {row['skin_tone']}" if row['skin_tone'] else "",
        f"with {row['skin_blemishes']} skin features" if row['skin_blemishes'] else "",
        f"{row['wrinkles']} wrinkles or age lines" if row['wrinkles'] else "",
        f"wearing {row['clothing']}" if row['clothing'] else "",
        f"accessories include {row['accessories']}" if row['accessories'] else "",
        f"glasses described as {row['glasses_description']}" if row['glasses_description'] else "",
        f"Notable standout features include {row['standout_features']}" if row['standout_features'] else "",
        f"The confidence level in this description is {row['confidence_level']}" if row['confidence_level'] else ""
    ]

    description = ""
    for part in filter(None, parts):
        # Add ". " only if the sentence doesn't end with certain conjunctions
        if part.startswith(("The")):
            description += ". " + part
        else:
            description += " " + part

    # Remove leading ". " and add final period
    description = description.strip(". ") + "."

    # Update the suspect_descriptions table with the generated summary
    cursor.execute('''
        UPDATE suspect_descriptions
        SET summary = ?
        WHERE user_id = ?
    ''', (description, user_id))

    get_db().commit()

    return description

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route to render the image upload page
@app.route('/upload')
def upload():
    user_id = request.args.get('userId')  # Get userId from query parameters
    if not user_id:
        return "User ID is missing!", 400
    return render_template('upload.html', user_id=user_id)

@app.route('/generated_images/<filename>')
def generated_file(filename):
    print(f"Attempting to serve file: {filename}")  # Debugging log
    print(f"Serving files from: {app.config['GENERATED_FOLDER']}")
    return send_from_directory(app.config['GENERATED_FOLDER'], filename)


# Route for uploading an image
@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        user_id = request.form.get('userId')
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        db = get_db()
        cursor = db.cursor()
        cursor.execute('''
            UPDATE suspect_descriptions
            SET uploaded_image = ?
            WHERE user_id = ?
        ''', (filename, user_id))
        db.commit()

        # Generate initial images on upload
        summary_text = cursor.execute(
            'SELECT summary FROM suspect_descriptions WHERE user_id = ?', (user_id,)
        ).fetchone()['summary']
        generated_images = run_clip_model(filepath, summary_text)
        generated_image_paths = save_generated_images(generated_images, user_id)

        return jsonify({
            'success': True,
            'message': 'Image uploaded and initial images generated successfully!',
            'generated_images': generated_image_paths
        }), 200

    return jsonify({'success': False, 'message': 'Invalid file type'}), 400

# Route to refine an image with a user-selected prompt and selected image
@app.route('/refine-image', methods=['POST'])
def refine_image():
    data = request.get_json()
    user_id = data.get("user_id")
    selected_image = data.get("selected_image")
    prompt_text = data.get("prompt")

    if not user_id or not selected_image or not prompt_text:
        return jsonify({"success": False, "message": "Missing data"}), 400

    # Generate new images based on the selected image and prompt
    refined_images = run_clip_model(selected_image, prompt_text)

    # Save the refined images
    refined_image_paths = save_generated_images(refined_images, user_id)

    # Return only the new images to the frontend
    return jsonify({"success": True, "images": refined_image_paths})


# Function to generate mock images using PIL based on an image and text prompt
def run_clip_model(image_path, text):
    """
    Simulate image generation by creating random dummy images with Pillow.
    """
    images = []
    for i in range(4):  # Generate 4 images
        img = Image.new('RGB', (256, 256), color=(255, 255, 255))  # White background
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"Image {i+1}\n", fill=(0, 0, 0))  # Add mock text
        images.append(img)
    return images

# Function to save generated images with incrementing filenames
def save_generated_images(images, user_id):
    saved_image_paths = []

    # Find the next available number for the user
    existing_files = glob.glob(f"{app.config['GENERATED_FOLDER']}/generated_image_{user_id}_*.png")
    existing_numbers = [
        int(os.path.splitext(os.path.basename(file))[0].split("_")[-1]) for file in existing_files
    ]
    next_number = max(existing_numbers, default=0) + 1

    for i, img in enumerate(images, start=next_number):
        filename = f"generated_image_{user_id}_{i}.png"
        filepath = os.path.join(app.config['GENERATED_FOLDER'], filename)
        img.save(filepath)  # Save the Pillow image
        saved_image_paths.append(filename)

    return saved_image_paths

@app.route('/generate-images', methods=['GET'])
def display_generated_images():
    user_id = request.args.get('userId')
    if not user_id:
        return jsonify({"success": False, "message": "User ID is missing!"}), 400

    # Fetch generated images for the user from the GENERATED_FOLDER
    generated_folder = app.config['GENERATED_FOLDER']
    generated_images = sorted(glob.glob(f"{generated_folder}/generated_image_{user_id}_*.png"))

    if not generated_images:
        return jsonify({"success": False, "message": "No generated images found!"}), 404

    # Log the images being returned
    generated_images = [os.path.basename(image) for image in generated_images]
    print(f"Returning generated images for user {user_id}: {generated_images}")  # Debug log

    return render_template('generated_images.html', user_id=user_id)

@app.route('/api/generate-images', methods=['GET'])
def get_generated_images():
    user_id = request.args.get('userId')
    if not user_id:
        return jsonify({"success": False, "message": "User ID is missing!"}), 400

    generated_folder = app.config['GENERATED_FOLDER']
    generated_images = sorted(glob.glob(f"{generated_folder}/generated_image_{user_id}_*.png"))
    latest_images = generated_images[-4:]  # Get the last 4 images

    if not latest_images:
        return jsonify({"success": False, "message": "No generated images found!"}), 404

    # Log the images being returned
    print(f"Returning generated images for user {user_id}: {generated_images}")  # Debug log

    latest_images = [os.path.basename(image) for image in latest_images]
    return jsonify({"success": True, "images": latest_images})


if __name__ == '__main__':
    init_db()
    app.run(debug=True)
from app import app
# app.py
from flask import Flask, request, jsonify, render_template, redirect, g, url_for
import sqlite3

DATABASE = 'responses.db'

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
        # age = data['age']
        # date_of_birth = data['date_of_birth']

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



if __name__ == '__main__':
    init_db()
    app.run(debug=True)
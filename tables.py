import sqlite3

DATABASE = 'responses.db'

def init_db():
    with sqlite3.connect(DATABASE) as db:
        cursor = db.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_info (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS survey_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL UNIQUE,
                question_id INTEGER NOT NULL,
                response TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES user_info(user_id)
            )
        ''')
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
    print("Database initialized successfully.")

if __name__ == '__main__':
    init_db()

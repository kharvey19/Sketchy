import os
import json

# Function to generate paragraph description for each entry
def generate_description(row):
    parts = [
        f"The suspect is described as {row['Gender']}" if row.get('Gender') and row['Gender'] != "None" else "",
        f"around {row['ApproximateAge']} years old" if row.get('ApproximateAge') and row['ApproximateAge'] != "None" else "",
        f"with a {row['BodyBuild']} build" if row.get('BodyBuild') and row['BodyBuild'] != "None" else "",
        f"approximately {row['Height']} tall" if row.get('Height') and row['Height'] != "None" else "",
        f"with distinctive features like {row['DistinctiveFeatures']}" if row.get('DistinctiveFeatures') and row['DistinctiveFeatures'] != "None" else "",
        f"The face shape is {row['FaceShape']}" if row.get('FaceShape') and row['FaceShape'] != "None" else "",
        f"and the hairstyle is {row['HairStyleAndTexture']}" if row.get('HairStyleAndTexture') and row['HairStyleAndTexture'] != "None" else "",
        f"with a {row['Forehead']} forehead" if row.get('Forehead') and row['Forehead'] != "None" else "",
        f"and {row['Eyebrows']} eyebrows" if row.get('Eyebrows') and row['Eyebrows'] != "None" else "",
        f"The eyes are {row['Eyes']}" if row.get('Eyes') and row['Eyes'] != "None" else "",
        f"with a {row['Nose']} nose" if row.get('Nose') and row['Nose'] != "None" else "",
        f"a {row['Mouth']} mouth" if row.get('Mouth') and row['Mouth'] != "None" else "",
        f"and {row['Teeth']} teeth" if row.get('Teeth') and row['Teeth'] != "None" else "",
        f"The jawline is {row['Jawline']}" if row.get('Jawline') and row['Jawline'] != "None" else "",
        f"with {row['FacialHair']} facial hair" if row.get('FacialHair') and row['FacialHair'] != "None" else "",
        f"The skin tone is {row['SkinTone']}" if row.get('SkinTone') and row['SkinTone'] != "None" else "",
        f"wearing {row['ClothingStyleAndTexture']}" if row.get('ClothingStyleAndTexture') and row['ClothingStyleAndTexture'] != "None" else "",
        f"accessories include {row['Accessories']}" if row.get('Accessories') and row['Accessories'] != "None" else "",
        f"Notable standout features include {row['NotableStandoutFeatures']}" if row.get('NotableStandoutFeatures') and row['NotableStandoutFeatures'] != "None" else "",
        f"The confidence level in this description is {row['ConfidenceLevel']}" if row.get('ConfidenceLevel') and row['ConfidenceLevel'] != "None" else ""
    ]
    
    description = " ".join(filter(None, parts)).strip()
    return description

# Function to create text files for each image description
def create_descriptions_from_json(json_file_path, output_folder):
    # Read the JSON data from the file
    with open(json_file_path, "r") as file:
        json_data = json.load(file)
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each entry in the JSON data
    for entry in json_data:
        file_name = entry.get("ImageFileName", "unknown").replace(".jpg", ".txt").replace(".jpeg", ".txt").replace(".png", ".txt")
        file_path = os.path.join(output_folder, file_name)
        
        # Generate the description
        description = generate_description(entry)
        
        # Write to a text file
        with open(file_path, "w") as file:
            file.write(description)
        print(f"Created description for {entry.get('ImageFileName', 'unknown')}")

# Main execution
if __name__ == "__main__":
    # Path to the JSON file
    json_file_path = "/Users/katherineharvey/Desktop/policesketch/image_analysis.json"  # Replace with the actual path to your JSON file
    
    # Specify output folder
    output_folder = "descriptions"
    
    # Generate descriptions
    create_descriptions_from_json(json_file_path, output_folder)

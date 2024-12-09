import os

# Directory containing your images
image_directory = "/Users/katherineharvey/Desktop/policesketch/chatgpt_descriptions/original_sketch"

# Output file to save image IDs or paths
output_file = "image_list.txt"

# List all image files and sort them alphabetically
image_files = sorted(f for f in os.listdir(image_directory) if f.endswith(('.jpg', '.jpeg', '.png')))

# Save the sorted list to a file
with open(output_file, "w") as f:
    for image_file in image_files:
        f.write(image_file + "\n")

print(f"Image list saved to {output_file}, sorted alphabetically.")

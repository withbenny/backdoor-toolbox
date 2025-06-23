import os
from PIL import Image

def resize_images_in_folder(folder_path, original_size=32, new_size=96):
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return
    print(f"Scanning folder: '{folder_path}'\n")


    for filename in os.listdir(folder_path):
        if filename.endswith(f"_{original_size}.png"):
            
            original_file_path = os.path.join(folder_path, filename)
            
            try:
                with Image.open(original_file_path) as img:
                    print(f"Find target file: '{filename}' (Original size: {img.size})")
                    
                    size = (new_size, new_size)
                    resized_img = img.resize(size, Image.Resampling.LANCZOS)
                    new_filename = filename.replace(f"_{original_size}.png", f"_{new_size}.png")
                    new_file_path = os.path.join(folder_path, new_filename)
                    
                    resized_img.save(new_file_path)
                    print(f"-> Successfully resized and saved as '{new_filename}' (New size: {resized_img.size})")

            except Exception as e:
                print(f"Processing file '{filename}' failed. Error: {e}\n")

    print("Done")

if __name__ == "__main__":
    target_folder = "triggers"
    resize_images_in_folder(target_folder, (96, 96))
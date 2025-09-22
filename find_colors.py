import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def find_unique_colors(directory):
    """Scans all PNG images in a directory and returns a set of unique (R, G, B) colors."""
    unique_colors = set()
    print(f"Scanning directory: {directory}")
    file_list = [f for f in os.listdir(directory) if f.endswith('.png')]

    if not file_list:
        print("No .png files found in this directory.")
        return unique_colors

    for filename in tqdm(file_list, desc="Analyzing masks"):
        path = os.path.join(directory, filename)
        try:
            img = Image.open(path).convert('RGB')
            # Reshape to a list of pixels [(R,G,B), (R,G,B), ...] and find unique rows
            img_array = np.array(img).reshape(-1, 3)
            colors = np.unique(img_array, axis=0)
            for color in colors:
                unique_colors.add(tuple(color))
        except Exception as e:
            print(f"Could not process {filename}: {e}")
    return unique_colors

if __name__ == "__main__":
    # --- 重要 ---
    # 将此路径指向您 *原始的、未经转换的* RGB标签目录
    label_dir = "/data/DeepGlobe_land_cover_ISDNet/rgb2id/val"
    # ------------

    if not os.path.isdir(label_dir):
        print(f"Error: Directory not found at {label_dir}")
    else:
        found_colors = find_unique_colors(label_dir)
        print("\n--- Found Unique Colors (R, G, B) ---")
        if not found_colors:
            print("No colors found. Is the directory correct?")
        else:
            # Print sorted colors for consistency
            for color in sorted(list(found_colors)):
                print(color)
        print("---------------------------------------")
        print("Please use this list to correct the COLOR_TO_ID_MAP in your convert_masks.py script.")

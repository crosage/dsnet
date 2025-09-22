import os
from PIL import Image
from tqdm import tqdm
import multiprocessing

def convert_image(tif_path):
    """Converts a single TIFF image to PNG."""
    png_path = os.path.splitext(tif_path)[0] + '.png'
    if not os.path.exists(png_path):
        try:
            with Image.open(tif_path) as img:
                img.save(png_path, 'PNG')
            os.remove(tif_path) # Optionally delete the original .tif to save space
        except Exception as e:
            return f"Failed to convert {tif_path}: {e}"
    return None

def convert_folder(source_dir):
    """Converts all TIFF images in a directory to PNG."""
    print(f"\nScanning directory for images: {source_dir}")
    tasks = [os.path.join(root, file) for root, _, files in os.walk(source_dir) 
             for file in files if file.lower().endswith('_sat.tif')]

    if not tasks:
        print("No '_sat.tif' files found to convert.")
        return

    print(f"Found {len(tasks)} images. Starting conversion...")
    with multiprocessing.Pool() as pool:
        list(tqdm(pool.imap_unordered(convert_image, tasks), total=len(tasks), desc="Converting Images"))
    print("Image conversion complete.")

if __name__ == '__main__':
    data_root = '/data/Inria_New_Structure'
    convert_folder(os.path.join(data_root, 'imgs'))
    print("\nAll image conversions finished!")

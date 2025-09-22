import os
import shutil
from PIL import Image
from tqdm import tqdm
import multiprocessing

def process_file(task):
    """Processes a single file: renames image or converts/renames label."""
    file_path, mode = task

    try:
        base_name = os.path.basename(file_path).replace('.tif', '')
        dir_name = os.path.dirname(file_path)

        if mode == 'image':
            # For images: rename xxx.tif to xxx_sat.tif
            new_name = f"{base_name}_sat.tif"
            new_path = os.path.join(dir_name, new_name)
            if not os.path.exists(new_path):
                shutil.move(file_path, new_path)

        elif mode == 'label':
            # For labels: convert xxx.tif to xxx_mask.png
            new_name = f"{base_name}_mask.png"
            new_path = os.path.join(dir_name, new_name)
            if not os.path.exists(new_path):
                with Image.open(file_path) as img:
                    # Ensure it's single-channel 8-bit
                    if img.mode != 'L':
                        img = img.convert('L')
                    img.save(new_path, 'PNG')
                os.remove(file_path) # Delete the old .tif label
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def prepare_directory(directory, mode):
    """Prepare all files in a directory using multiple processes."""
    print(f"\n--- Preparing directory: {directory} (Mode: {mode}) ---")

    tasks = []
    for filename in os.listdir(directory):
        if filename.lower().endswith('.tif'):
            tasks.append((os.path.join(directory, filename), mode))

    if not tasks:
        print("No .tif files found to process.")
        return

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(process_file, tasks), total=len(tasks), desc=f"Processing {mode}s"))

    print("Done.")

if __name__ == '__main__':
    data_root = '/data/Inria_New_Structure'

    # Process images (rename)
    prepare_directory(os.path.join(data_root, 'imgs/train'), 'image')
    prepare_directory(os.path.join(data_root, 'imgs/val'), 'image')

    # Process labels (convert and rename)
    prepare_directory(os.path.join(data_root, 'labels/train'), 'label')
    prepare_directory(os.path.join(data_root, 'labels/val'), 'label')

    print("\nInria data preparation complete!")

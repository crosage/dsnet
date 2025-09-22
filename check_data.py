from PIL import Image
import numpy as np

try:
    # We will test one image-label pair
    img_path = '/data/Inria_New_Structure/imgs/val/austin31.tif'
    label_path = '/data/Inria_New_Structure/labels/val/austin31.tif'

    img = Image.open(img_path)
    label = Image.open(label_path)
    
    print("✅ Success! Opened both image and label files.")
    print("---")
    print(f"Image details -> Path: {img_path} | Mode: {img.mode} | Size: {img.size}")
    print(f"Label details -> Path: {label_path} | Mode: {label.mode} | Size: {label.size}")
    print("---")

    # This is the most important check
    if label.mode in ['L', 'P', '1']:
        print("✅ The label file appears to be a valid single-channel mask.")
        # Check unique pixel values in the label
        label_pixels = np.unique(np.array(label))
        print(f"Unique pixel values in label: {label_pixels}")
        if len(label_pixels) <= 2:
            print("This looks like a correct binary (0 and 1) mask.")
        else:
            print("This mask has multiple classes.")

    else:
        print("❌ CRITICAL ERROR: The label file is not a valid mask!")
        print(f"The label's mode is '{label.mode}', which is likely an RGB color image.")
        print("Segmentation labels MUST be single-channel grayscale images (Mode 'L' or 'P').")

except FileNotFoundError:
    print("❌ Error! A file was not found. Check the paths.")
except Exception as e:
    print("❌ Error! An unexpected error occurred:", e)

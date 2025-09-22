from PIL import Image
import numpy as np

# --- 请修改这里 ---
# 1. 把下面的文件名换成您DeepGlobe标签目录下一个真实存在的文件名
#    您可以运行 ls /data/DeepGlobe_land_cover_ISDNet/rgb2id/train 来找一个文件名
label_filename = "100034_mask.png" # 这是一个示例，请务必替换

# 2. 这是您的标签文件所在的目录
label_dir = "/data/DeepGlobe_land_cover_ISDNet/rgb2id/train"
# ---------------------

image_path = f"{label_dir}/{label_filename}"

try:
    print(f"--- 正在检查文件: {image_path} ---")
    img = Image.open(image_path)

    # 1. 检查图像模式 (Image Mode)
    #    - 'RGB' 表示三通道彩色图 (这是我们怀疑的问题)
    #    - 'L' 或 'P' 表示单通道索引图 (这是正确的格式)
    print(f"图像模式 (Image Mode): {img.mode}")

    # 2. 检查图像形状 (Shape)
    img_array = np.array(img)
    #    - (高, 宽, 3) 表示三通道 (有问题)
    #    - (高, 宽) 表示单通道 (正确)
    print(f"图像形状 (Shape): {img_array.shape}")

    # 3. 检查像素值 (Pixel Values)
    #    - 如果是单通道ID图，这里会显示出所有的类别ID，比如 [0 1 2 3 4 5 6]
    #    - 如果是三通道RGB图，这个列表可能会很长，包含0-255之间的很多值
    unique_values = np.unique(img_array)
    print(f"图中包含的独立像素值 (Unique Values): {unique_values[:20]} ... (最多显示前20个)")

except FileNotFoundError:
    print(f"\n错误：文件未找到！")
    print(f"请确认文件名 '{label_filename}' 是否真实存在于 '{label_dir}' 目录中。")
except Exception as e:
    print(f"\n发生未知错误: {e}")

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def convert_masks(source_dir, target_dir, color_map):
    """
    将源目录中的RGB掩码图像根据颜色映射转换为单通道ID图，并保存到目标目录。
    """
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 获取所有待处理的文件名
    filenames = os.listdir(source_dir)
    
    print(f"找到 {len(filenames)} 个掩码文件，开始转换...")
    
    # 使用tqdm显示进度条
    for filename in tqdm(filenames, desc="转换进度"):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, filename)
            
            try:
                # 打开RGB图像
                rgb_mask = Image.open(source_path).convert('RGB')
                rgb_array = np.array(rgb_mask)
                
                # 获取图像尺寸
                height, width, _ = rgb_array.shape
                
                # 创建一个用于存放ID的单通道数组，默认为背景ID
                # 注意：如果您的数据集中没有(0,0,0)的背景，可以设为一个不会使用的值，如255
                id_array = np.full((height, width), fill_value=6, dtype=np.uint8)
                
                # 根据颜色映射表进行转换
                for color, class_id in color_map.items():
                    # 找到所有匹配当前颜色的像素位置
                    matches = np.where(np.all(rgb_array == color, axis=-1))
                    # 在这些位置上，将值设为对应的class_id
                    id_array[matches] = class_id
                
                # 将Numpy数组转换回Pillow图像
                id_mask = Image.fromarray(id_array, mode='L') # 'L'模式为8位单通道灰度图
                
                # 保存新的单通道掩码图
                id_mask.save(target_path)

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")

if __name__ == '__main__':
    # --- 1. 定义颜色->ID映射关系 ---
    # 格式: (R, G, B): class_id
    # 请根据您的数据集的实际情况进行调整
    COLOR_TO_ID_MAP = {
        (0, 0, 0):       0,  # unknown
        (0, 255, 255):   1,  # urban
        (255, 255, 0):   2,  # agriculture
        (255, 0, 255):   3,  # rangeland
        (0, 255, 0):     4,  # forest
        (0, 0, 255):     5,  # water
        (255, 255, 255): 6   # barren
    }

    # --- 2. 定义源目录和目标目录 ---
    # 源目录：存放原始RGB标签图的地方
    # 目标目录：存放转换后的单通道ID图的地方
    data_root = '/data/DeepGlobe_land_cover_ISDNet'
    
    # 处理训练集
    print("\n--- 处理训练集 ---")
    train_source = os.path.join(data_root, 'rgb2id/train')
    train_target = os.path.join(data_root, 'ann_dir/train') # 新建一个目录存放正确的标签
    convert_masks(train_source, train_target, COLOR_TO_ID_MAP)

    # 处理验证集
    print("\n--- 处理验证集 ---")
    val_source = os.path.join(data_root, 'rgb2id/val')
    val_target = os.path.join(data_root, 'ann_dir/val')
    convert_masks(val_source, val_target, COLOR_TO_ID_MAP)

    print("\n转换完成！新的标签已保存在 'ann_dir' 目录下。")
    print("下一步：请修改您的配置文件，将 'ann_dir' 指向新的目录。")

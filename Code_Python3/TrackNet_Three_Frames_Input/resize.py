from PIL import Image
import os

input_folder = 'imgs'
output_folder = 'output'
target_size = (320, 128)

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取输入文件夹中的所有图片文件
image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

# 遍历每个图片文件并调整大小
for file_name in image_files:
    input_path = os.path.join(input_folder, file_name)
    output_path = os.path.join(output_folder, file_name)

    # 打开图像文件
    input_image = Image.open(input_path)

    # 调整大小
    resized_image = input_image.resize(target_size)

    # 保存调整后的图像文件
    resized_image.save(output_path)

    print(f"Resized {file_name} and saved to {output_path}")

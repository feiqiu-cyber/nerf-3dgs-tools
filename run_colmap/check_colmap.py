import os
from read_write_model import read_images_binary, read_cameras_binary

def get_unregistered_images(images_bin_path, images_dir):
    """
    检查 images.bin 文件中是否包含图像目录中的所有图像，并返回未注册的图像列表。

    :param images_bin_path: images.bin 文件的路径
    :param images_dir: 图像目录的路径
    :return: 未注册的图像列表
    """
    # 读取 images.bin 文件
    images = read_images_binary(images_bin_path)
    
    # 获取已注册图像的名称
    registered_images = {image.name for image in images.values()}
    
    # 获取图像目录中的所有图像文件名
    all_images = {file for file in os.listdir(images_dir) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'))}
    
    # 找出未注册的图像
    unregistered_images = all_images - registered_images
    
    return list(unregistered_images), list(registered_images)

# 示例用法
images_bin_path = '/cfs/wangboyuan/dataset/indoor/office/sparse/1/images.bin'
images_dir = '/cfs/wangboyuan/dataset/indoor/office//images'

# 获取未注册的图像列表
unregistered_images, registered_images = get_unregistered_images(images_bin_path, images_dir)

if not unregistered_images:
    print("All images are registered in images.bin.")
else:
    print("The following images are unregistered in images.bin:")
    for img in unregistered_images:
        print(img)
    print(f"total num: {len(unregistered_images)}")

camera_bin = '/cfs/wangboyuan/workspace/Scaffold-GS/data/indoor/office/sparse/cameras.bin'
cameras = read_cameras_binary(camera_bin)
print(cameras)
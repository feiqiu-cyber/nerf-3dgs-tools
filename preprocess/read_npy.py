import matplotlib.pyplot as plt
import numpy as np
import cv2

import os

def save_normal_map_as_png(npy_file_path, output_png_path):
    """
    读取 .npy 文件中的法线数据，并将其保存为 .png 图像。

    参数:
    npy_file_path (str): .npy 文件的路径
    output_png_path (str): 输出的 .png 文件的路径
    """
    # 读取 .npy 文件
    normals = np.load(npy_file_path)

    # 检查数据形状，应该是 (1, 3, 1080, 1920)
    if normals.shape[0] != 1 or normals.shape[1] != 3:
        raise ValueError("Expected input shape to be (1, 3, 1080, 1920)")

    # 移除第一个维度 (batch dimension)
    normals = normals[0]

    # 将法线数据调整为 [0, 1] 之间，以便可视化
    normals_visual = (normals + 1) / 2

    # 转换为 (1080, 1920, 3) 以便使用 matplotlib 可视化
    normals_visual = np.transpose(normals_visual, (1, 2, 0))

    # 使用 matplotlib 保存为 .png
    plt.imsave(output_png_path, normals_visual)

def process_folder(input_folder, output_folder):
    """
    处理输入文件夹中的所有 .npy 文件，并将输出保存到另一个文件夹中。

    参数:
    input_folder (str): 输入文件夹的路径，包含 .npy 文件
    output_folder (str): 输出文件夹的路径，用于保存 .png 文件
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有 .npy 文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.npy'):
            npy_file_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + '.png'
            output_png_path = os.path.join(output_folder, output_filename)
            
            # 保存法线数据为 .png 文件
            save_normal_map_as_png(npy_file_path, output_png_path)
            print(f"Saved {output_png_path}")


def edge_detection(image_path, method='canny', low_threshold=50, high_threshold=150):
    """
    对给定图像进行边缘检测，并返回边缘检测结果。

    参数:
    image_path (str): 输入图像的路径
    method (str): 使用的边缘检测方法 ('sobel' 或 'canny')
    low_threshold (int): 低阈值 (用于 Canny 边缘检测)
    high_threshold (int): 高阈值 (用于 Canny 边缘检测)

    返回:
    edges (numpy.ndarray): 边缘检测结果
    """
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if method == 'sobel':
        # 使用 Sobel 算法进行边缘检测
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        edges = cv2.magnitude(grad_x, grad_y)
    elif method == 'canny':
        # 使用 Canny 算法进行边缘检测
        edges = cv2.Canny(image, low_threshold, high_threshold)
    else:
        raise ValueError("Unsupported method. Use 'sobel' or 'canny'.")

    return edges

def save_edge_image(edges, output_path):
    """
    保存边缘检测结果为图像文件。

    参数:
    edges (numpy.ndarray): 边缘检测结果
    output_path (str): 输出图像文件的路径
    """
    plt.imsave(output_path, edges, cmap='gray')

def process_folder_for_edges(input_folder, output_folder, method='canny', low_threshold=50, high_threshold=150):
    """
    处理输入文件夹中的所有图像文件，并将边缘检测结果保存到输出文件夹中。

    参数:
    input_folder (str): 输入文件夹的路径，包含图像文件
    output_folder (str): 输出文件夹的路径，用于保存边缘检测结果
    method (str): 使用的边缘检测方法 ('sobel' 或 'canny')
    low_threshold (int): 低阈值 (用于 Canny 边缘检测)
    high_threshold (int): 高阈值 (用于 Canny 边缘检测)
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 支持的图像文件扩展名
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    # 遍历输入文件夹中的所有图像文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_extensions):
            input_image_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + '_edges.png'
            output_image_path = os.path.join(output_folder, output_filename)

            # 执行边缘检测
            edges = edge_detection(input_image_path, method, low_threshold, high_threshold)
            
            # 保存边缘检测结果
            save_edge_image(edges, output_image_path)
            print(f"Saved {output_image_path}")

if __name__ == "__main__":
    input_folder = '/cfs/wangboyuan/dataset/indoor/office/images'
    output_folder = '/cfs/wangboyuan/dataset/indoor/office/edge'
    # process_folder(input_folder, output_folder)
    process_folder_for_edges(input_folder, output_folder, method='canny', low_threshold=50, high_threshold=150)


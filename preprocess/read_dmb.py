import numpy as np

def read_dmb(file_path):
    with open(file_path, 'rb') as file:
        # 假设文件头包含数据尺寸
        width = np.fromfile(file, dtype=np.int32, count=1)[0]
        height = np.fromfile(file, dtype=np.int32, count=1)[0]
        
        # 假设深度数据存储在前景部分
        depth_data = np.fromfile(file, dtype=np.float32, count=width * height*3)

        
        return depth_data

# 示例使用
file_path = '/cfs/wangboyuan/dataset/indoor/office/mvs_input/ACMMP/2333_00000332/depths.dmb'
depth = read_dmb(file_path)

print("Depth Data:", depth.shape)

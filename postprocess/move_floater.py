import numpy as np
from plyfile import PlyData, PlyElement
import torch
from torch import nn
from errno import EEXIST
from os import makedirs, path
import os
from scipy.spatial import cKDTree

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def load_ply(path, max_sh_degree=3):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # _xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
    # _features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    # _features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    # _opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
    # _scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
    # _rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

    active_sh_degree = max_sh_degree
    # return {
    #     "xyz": _xyz,
    #     "feature_dc": _features_dc,
    #     "feature_rest": _features_rest,
    #     "scaling": _scaling,
    #     "rotation": _rotation
    # }
    return {
            "xyz": xyz,
            "feature_dc": features_dc,
            "feature_rest": features_extra,
            "scaling": scales,
            "rotation": rots
        }
    # return _xyz, _features_dc, _features_rest, _scaling, _rotation

def construct_list_of_attributes(_features_dc, _features_rest, _opacity, _scaling, _rotation):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(_features_dc.shape[1]*_features_dc.shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(_features_rest.shape[1]*_features_rest.shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(_scaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(_rotation.shape[1]):
        l.append('rot_{}'.format(i))
    return l

def save_ply(path, _xyz, _features_dc, _features_rest, _opacity, _scaling, _rotation ):
    mkdir_p(os.path.dirname(path))

    xyz = _xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = _features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = _features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = _opacity.detach().cpu().numpy()
    scale = _scaling.detach().cpu().numpy()
    rotation = _rotation.detach().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(_features_dc, _features_rest, _scaling, _rotation)]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def remove_sparse_points(point_cloud, radius, sparsity_threshold):
    """
    删除点云中稀疏的点。

    参数：
    - point_cloud: (N, 3) 的 numpy 数组，表示点云的 xyz 坐标
    - radius: float，表示邻域半径
    - sparsity_threshold: int，表示稀疏度阈值

    返回：
    - filtered_point_cloud: (M, 3) 的 numpy 数组，表示过滤后的点云
    """
    # 创建一个 k-d 树用于快速查询
    tree = cKDTree(point_cloud)
    
    # 计算每个点的邻域内点的数量
    counts = tree.query_ball_point(point_cloud, r=radius, return_length=True)
    
    # 筛选出不稀疏的点
    dense_points = point_cloud[counts >= sparsity_threshold]
    
    return dense_points



def calculate_sparsity(point_cloud, radius):
    """
    计算所有点的稀疏度。

    参数：
    - point_cloud: (N, 3) 的 numpy 数组，表示点云的 xyz 坐标
    - radius: float，表示邻域半径

    返回：
    - sparsity: (N,) 的 numpy 数组，表示每个点的稀疏度
    """
    # 创建一个 k-d 树用于快速查询
    print(point_cloud.shape)
    tree = cKDTree(point_cloud)
    
    # 计算每个点的邻域内点的数量
    counts = tree.query_ball_point(point_cloud, r=radius, return_length=True)
    
    # 计算邻域体积
    sphere_volume = (4/3) * np.pi * radius**3
    
    # 计算每个点的稀疏度
    # sparsity = counts / sphere_volume
    sparsity = counts
    
    return sparsity


if __name__ == "__main__":
    path = '/cfs/wangboyuan/workspace/gaussian-splatting/output/7c5a86c0-8/point_cloud/iteration_30000/point_cloud.ply'
    result = load_ply(path)
    xyz = result['xyz']
    spa = calculate_sparsity(xyz, 10)
    print(spa.size())
    print(np.sum(spa < 10))
    

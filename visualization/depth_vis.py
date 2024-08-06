import numpy as np
import glob
import matplotlib
import cv2

import os

npy_dir = '/cfs/wangboyuan/dataset/indoor/office/depths'
output_dir = npy_dir.replace("depths", "depths_vis")
cmap = matplotlib.colormaps.get_cmap('Spectral_r')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for i in glob.glob(npy_dir + "/*.npy"):
    npy_path = os.path.join(npy_dir, i)
    depth = np.load(npy_path)
    depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, os.path.splitext(os.path.basename(npy_path))[0] + '.png'), depth)
"""
Author: Min Seok Lee and Wooseok Shin
TRACER: Extreme Attention Guided Salient Object Tracing Network
git repo: https://github.com/Karel911/TRACER
"""
import os
import cv2
import numpy as np
from tqdm import tqdm

# Append custom datasets below list
dataset_list = ['DUTS', 'DUT-O', 'HKU-IS', 'ECSSD', 'PASCAL-S']


def edge_generator(dataset):
    if dataset == 'DUTS':
        mask_path = os.path.join('data/', dataset, 'Train/masks/')
    else:
        mask_path = os.path.join('data/', dataset, 'Test/masks/')
    save_path = os.path.join('data/', dataset, 'Train/edges/')
    os.makedirs(save_path, exist_ok=True)
    mask_list = os.listdir(mask_path)

    for i, img_name in tqdm(enumerate(mask_list)):
        mask = cv2.imread(mask_path + img_name)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = np.int64(mask > 128)

        [gy, gx] = np.gradient(mask)
        tmp_edge = gy * gy + gx * gx
        tmp_edge[tmp_edge != 0] = 1
        bound = np.uint8(tmp_edge * 255)
        cv2.imwrite(os.path.join(save_path, f'{img_name}'), bound)


if __name__ == '__main__':
    for dataset in dataset_list:
        edge_generator(dataset)
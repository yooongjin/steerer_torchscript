
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
from PIL import Image
import math
import os
import pprint
import _init_paths
from lib.core.Counter import Counter
from lib.utils.utils import create_logger, random_seed_setting
from lib.utils.modelsummary import get_model_summary
from lib.core.cc_function import test_cc
import datasets
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import numpy as np
import timeit
import time
import logging
import argparse
from lib.models.build_counter import Baseline_Counter
from lib.utils.dist_utils import (
    get_dist_info,
    init_dist)
from mmcv import Config, DictAction
import einops
import cv2
import tqdm
import matplotlib.pyplot as plt
import pickle
import glob


def parse_args():
    parser = argparse.ArgumentParser(description='Test crowd counting network')
    parser.add_argument("--checkpoint", type=str,
                        required=True,
                        help="model checkpoint")
    parser.add_argument("--image", type=str,
                        required=True,
                        help="image path")
    parser.add_argument('--save_path', type=str,
                        default='')
    args = parser.parse_args()


    return args


def main():
    args = parse_args()
    model = torch.jit.load(args.checkpoint)
    time_res=[]
    pred_results = []
    model = model.cuda()
    model.eval()
    img_path = args.image
    with torch.no_grad():
        cv_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        height, width, _ = cv_image.shape
        cv_image = cv2.resize(cv_image, dsize=(width//4, height//4))

        input_image = torch.from_numpy(cv_image).float()
        input_image = input_image.unsqueeze(0).cuda()
        start_time = time.time()
        result = model(input_image)
        time_res.append(time.time() - start_time)
        pred = result
        pred_cnt = torch.sum(pred).item()
        pred_results.append(pred_cnt)

        
        pred = pred.squeeze().detach().cpu().numpy()
        max_pred = np.max(pred)
        min_pred = np.min(pred)
        
        pred = (pred - min_pred) / (max_pred - min_pred)
        pred = pred * 255
        pred = pred.astype(np.uint8)
        
        pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)

        res = cv2.addWeighted(cv_image, 0.6, pred, 0.4, 0)
        text = f"pred : {int(pred_cnt)}"
        org = (10, 40)  
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(res, text, org, font, 1, (0, 0, 255), 2) 
        cv2.imwrite(os.path.join(args.save_path, "test_"+os.path.basename(args.image)), res)


if __name__ == '__main__':
    main()

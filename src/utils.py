"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import torch
from PIL import Image
import numpy as np
import random


def cal_dice(prediction, label, num=2):
    """
    calculate each dice for all labels added by zxpan
    :param prediction: pred
    :param label: label
    :param num: class number
    :return: total dice
    """
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def dice(input, target, ignore_index=None):
    """
    calculate the whole dice added by zxpan
    :param input: pred
    :param target: label
    :param ignore_index: ignore index
    :return: dice
    """
    smooth = 1.
    # using clone, so that it can do change to original target.
    iflat = input.clone().view(-1)
    tflat = target.clone().view(-1)
    if ignore_index is not None:
        mask = tflat == ignore_index
        tflat[mask] = 0
        iflat[mask] = 0
    intersection = (iflat * tflat).sum()

    return (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = acc * 100

    return acc


def oht_to_scalar(y_pred):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    return y_pred_tags
    

def colorize_mask(mask, palette):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return np.array(new_mask.convert('RGB'))


def to_labels(masks, palette):
    results = np.zeros((len(masks), 256, 256), dtype=np.int32)
    label = 0
    for color in palette:
        idxs = np.where((masks == color).all(-1))
        results[idxs] = label
        label += 1
    return results
    

def setup_seed(seed):
    print('Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
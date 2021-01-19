#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/1/10 10:59
# @Author : lepold
# @File : utils.py
"""
    to prepare the image patch from our exist image data and divide into 3 groups,
    1) "group_a" : totally 50 camera models and promise 40k image patches each model,
    2) "group_b" : totally 30 camera models and each has at least 25k patches,
    3) "group_c" : the remaining patches from other camera models.
"""

# prepare data: move all images to a dir and split by camera models.
# e.g, "root dir" + "huawei" + "image_patch_0",
#                   "iphone" + "image_patch_0".

import math
import multiprocessing
import os
import traceback
import shutil
import numpy as np
from PIL import Image


def organize_image_path():
    pass


# STEP 3
# In the original paper, there exists three task: preprocess, camera model compare and edit log compare.
# So, we provide three dir for the three tasks respectively, as the paper said.


def partion_data():
    """
    preprocess, i.e, train a feature extractor: totally 50 models * 40k patches = 2000k patches from 'group_a'
    camera model compare:
                1) train: 1000k pair of image patches from 'group_a' and 'group_b', i think all the models
                50 + 30 models should be considered
                2) test: 1200k patches from 'group_a' and 'group_c'
    edit log compare:
                1) train: 400k patches from "a" and "b", after the tampering process, provided to similarity  network,
                50% percent is belonging to the same camera model, and remaining 50% is from different models.
                2) test: 1200k patches from 'group_a' and 'group_c', 10 from group_a and 15 from group_c.

    """
    pass


def printClasses(root):
    models = os.listdir(root)
    models.remove('list.txt')
    for idx, model in enumerate(models):
        print(idx, model)


def make_txt_file(root):
    """
    Generate a TXT file with the relative pathes of each image
    example:
    \model1\img1
    \model1\img2
    \model2\img1
    ...

    Args:
        root: the root path of img dataset

    """
    models = os.listdir(root)
    path_str = []
    flag = True
    try:
        models.remove('list.txt')
    except:
        traceback.print_exc()
    for index, model in enumerate(models):
        imgs = os.listdir(os.path.join(root, model))
        for img in imgs:
            if flag:
                flag = False
            else:
                path_str.append('\n')
            path_str.append(os.path.join(model, img) + '&{:02d}'.format(index))

    with open(os.path.join(root, 'list.txt'), 'w') as f:
        f.writelines(path_str)
        f.close()


def imgSlicer(file_path, patch_size=256, stride=256):
    """
    Cut an image into patches of patch size.

    Args:
        file_path: the location of the image
        patch_size: the size of a patch
        stride: step of cutting

    Returns:
        A list of patches.
        example:
        [patch1, patch2,....]
    """
    img = Image.open(file_path)
    img = np.array(img)
    h, w = img.shape[0], img.shape[1]
    patches = []
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = img[i:i + patch_size, j:j + patch_size]
            if patch is None:
                continue
            patches.append(patch)
    return patches


def createDataset(root, save_path, patch_size=256, stride=256, processes=40):
    model_paths = os.listdir(root)
    pool = multiprocessing.Pool(processes=processes)
    for model in model_paths:
        model_path = os.path.join(root, model)
        imgs = os.listdir(model_path)
        save_model_path = os.path.join(save_path, model)
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        length = len(imgs)
        num_processes = 4
        try:
            processes = []
            for i in range(num_processes):
                # p = multiprocessing.Process(
                #     target=multiImgSlicer,
                #     args=('process{:02d}'.format(i+1), patch_size, model_path, imgs[math.floor(i / num_processes * length):math.floor((i+1) / num_processes * length)], save_model_path))
                # processes.append(p)
                pool.apply_async(
                    imgSlicerProcess,
                    ('process{:02d}'.format(i + 1), patch_size, stride,
                     model_path,
                     imgs[math.floor(i / num_processes * length):math.floor(
                         (i + 1) / num_processes * length)], save_model_path))
            # for p in processes:
            #     p.start()
            #     # p.join()
        except:
            print("Error: 无法启动进程！")
            traceback.print_exc()
    pool.close()
    pool.join()


def imgSlicerProcess(thread_name, patch_size, stride, model_path, imgs,
                     save_model_path):
    print(thread_name + " start!")
    for img in imgs:
        img_path = os.path.join(model_path, img)
        patches = imgSlicer(img_path, patch_size=patch_size, stride=stride)
        for idx, patch in enumerate(patches):
            patch_img = Image.fromarray(patch.astype('uint8')).convert('RGB')
            patch_name = "{:s}_{:04d}.png".format(img[:-4], idx + 1)
            patch_img.save(os.path.join(save_model_path, patch_name))
    print(thread_name + " end!")


def copy(root, new_root, number=100000, processes=10):
    models = os.listdir(root)
    num_per_model = math.floor(number / len(models))
    pool = multiprocessing.Pool(processes=processes)
    models.remove('list.txt')
    for model in models:
        model_path = os.path.join(root, model)
        if not os.path.exists(os.path.join(new_root, model)):
            os.makedirs(os.path.join(new_root, model))
        imgs = os.listdir(model_path)
        indices = np.arange(len(imgs))
        np.random.shuffle(indices)
        # p = multiprocessing.Process(target=copyProcess,
        #                             args=(model, new_root, num_per_model, model,
        #                                   model_path, imgs, indices))
        # processes.append(p)
        pool.apply_async(
            copyProcess,
            (model, new_root, num_per_model, model, model_path, imgs, indices))
    # for p in processes:
    #     p.start()
    pool.close()
    pool.join()


def copyProcess(process_name, new_root, num_per_model, model, model_path, imgs,
                indices):
    print(process_name, "start coping!")
    for index in indices[:num_per_model]:
        try:
            shutil.copy(os.path.join(model_path, imgs[index]),
                        os.path.join(new_root, model, imgs[index]))
        except:
            traceback.print_exc()
    print(process_name, "end coping!")


# if __name__ == "__main__":
#     root = '../../data/sp-society-camera-model-identification/train/train'
#     save_path = '../../data/patches/phaseA/train128'
#     new_save_path = '../../data/patches/phaseA/train2'
#     createDataset(root=root, save_path=save_path, patch_size=256)
#     make_txt_file(save_path)
#     printClasses(save_path)
#     copy(save_path, new_save_path, number=100000)

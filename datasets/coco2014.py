# -*- coding: utf-8 -*-
# @Time    : 2019/6/27
# @Author  : Godder
# @Github  : https://github.com/WangGodder
from scipy import io as sio
import numpy as np
import os
from .loader import HashDataset
from . import transform_train, transform_test
default_img_mat_url = os.path.join( "./data", "coco2014", "imgList.mat")
default_tag_mat_url = os.path.join( "./data", "coco2014", "tagList.mat")
default_label_mat_url = os.path.join( "./data", "coco2014", "labelList.mat")
default_image_dir = os.path.join("./data", 'coco2014')
default_seed = 8

img_names = None
txt = None
label = None


def load_mat(img_mat_url=default_img_mat_url, tag_mat_url=default_tag_mat_url, label_mat_url=default_label_mat_url):
    img_names = sio.loadmat(img_mat_url)['imgs']  # type: np.ndarray
    img_names = img_names.squeeze()
    all_img_names = img_names
    all_txt = np.array(sio.loadmat(tag_mat_url)['tags'], dtype=np.float)
    all_label = np.array(sio.loadmat(label_mat_url)['labels'], dtype=np.float)
    return all_img_names, all_txt, all_label


def split_data(all_img_names, all_txt, all_label, query_num=5000, train_num=10000, seed=None):
    np.random.seed(seed)
    random_index = np.random.permutation(range(all_img_names.shape[0]))
    query_index = random_index[: query_num]
    train_index = random_index[query_num: query_num + train_num]
    retrieval_index = random_index[query_num:]

    query_img_names = all_img_names[query_index]
    train_img_names = all_img_names[train_index]
    retrieval_img_names = all_img_names[retrieval_index]

    query_txt = all_txt[query_index]
    train_txt = all_txt[train_index]
    retrieval_txt = all_txt[retrieval_index]

    query_label = all_label[query_index]
    train_label = all_label[train_index]
    retrieval_label = all_label[retrieval_index]

    img_names = (query_img_names, train_img_names, retrieval_img_names)
    txt = (query_txt, train_txt, retrieval_txt)
    label = (query_label, train_label, retrieval_label)
    return img_names, txt, label


def load_dataset(img_dir = default_image_dir, img_mat_url=default_img_mat_url, tag_mat_url=default_tag_mat_url,
                 label_mat_url=default_label_mat_url, train_num=10000, query_num=5000, seed=default_seed, **kwargs):
    global img_names, txt, label
    if img_names is None:
        all_img_names, all_txt, all_label = load_mat(img_mat_url, tag_mat_url, label_mat_url)
        img_names, txt, label = split_data(all_img_names, all_txt, all_label, query_num, train_num, seed)
        print("COCO2014 data load and shuffle by seed %d" % seed)

    query_dataset = HashDataset(image_dir= img_dir,images= img_names[0], texts= txt[0], labels= label[0] ,transform= transform_test)
    train_dataset = HashDataset(image_dir= img_dir,images= img_names[1], texts= txt[1], labels= label[1] ,transform= transform_train)
    gallery_dataset = HashDataset(image_dir= img_dir, images= img_names[2], texts= txt[2], labels= label[2], transform= transform_test)
    print(f"The number of query images is {len(query_dataset)}, the number of training images is {len(train_dataset)},"
          f" the number of gallery images is {len(gallery_dataset)} ")
    return query_dataset, train_dataset, gallery_dataset



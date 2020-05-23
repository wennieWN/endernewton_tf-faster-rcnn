# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import cv2
from model.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
import os
# import tensorflow as tf


def get_minibatch(roidb, num_classes):
  """Given a roidb, construct a minibatch sampled from it."""
  num_images = len(roidb)
  # Sample random scales to use for each image in this batch
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=num_images)
  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)

  # Get the input image blob, formatted for caffe
  im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

  blobs = {'data': im_blob}

  assert len(im_scales) == 1, "Single batch only"
  assert len(roidb) == 1, "Single batch only"

  # todo: get cls_filter blobs['clp_filter']
  # wn modified
  # 1. get file path
  sep = '/'
  clp_file_format = '.npy'
  clp_file_store = 'CloudPoints'

  img_path = roidb[0]['image']
  img_path_arr = img_path.split(sep)
  prefix = img_path_arr[:-2]
  file_name = img_path_arr[-1].split('.')[0] + clp_file_format
  clp_path = os.path.join(sep.join(prefix), clp_file_store, file_name)

  # 2. get cls data [?, 2]
  valid_points= np.load(clp_path)  # [?, 2]
  # todo: width & height is not fixed
  width_ori = roidb[0]['height']  # 322
  height_ori = roidb[0]['width']  # 500

  clp_ori = np.zeros([width_ori, height_ori], dtype=np.float32)  # 初始化
  clp_ori[tuple((valid_points.T[1, :], valid_points.T[0, :]))] = 1  # 设置存在点云的网格值为1 [322,500]

  # 3.resize cls [322,500] =》[600,932] （同图片的操作）
  clp_reshape = np.empty([width_ori, height_ori, 3], dtype=np.float32)
  for i in range(3):
    clp_reshape[0:width_ori, 0:height_ori, i] = clp_ori
  clp_res = cv2.resize(clp_reshape, None, None, fx=im_scales[0], fy=im_scales[0], interpolation=cv2.INTER_LINEAR)
  clp_res = clp_res[:, :, 0]  # [600,932]
  clp_res[clp_res > 0] = 1  # >0的值均设置成1

  width = clp_res.shape[0]
  height = clp_res.shape[1]
  clp_res = clp_res.reshape([1, width, height, 1])

  blobs['clp_info'] = clp_res  # [1,600,932,1]

  # 4. Max pooling
  # width = clp_res.shape[0]  # 600
  # height = clp_res.shape[1]  # 932
  # clp_res = clp_res.reshape([1, width, height, 1])
  # clp_filter = tf.constant(clp_res)
  # clp_filter_reshape = tf.reshape(clp_filter, [1, width, height, 1])
  #
  # clp_pooling = tf.nn.max_pool(clp_filter_reshape, [1, 16, 16, 1], [1, 16, 16, 1], padding='SAME') # self._feat_stride[0] = 16
  # clp_pooling = clp_pooling[0, :, :, 0]
  # print("pooling: " + str(clp_pooling.shape))
  # blobs['clp_filter'] = clp_pooling  # [38, 59] （同特征图net_conv尺寸一致）

  
  # gt boxes: (x1, y1, x2, y2, cls)
  if cfg.TRAIN.USE_ALL_GT:
    # Include all ground truth boxes
    gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
  else:
    # For the COCO ground truth boxes, exclude the ones that are ''iscrowd'' 
    gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
  gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
  gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
  gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
  blobs['gt_boxes'] = gt_boxes
  blobs['im_info'] = np.array(
    [im_blob.shape[1], im_blob.shape[2], im_scales[0]],
    dtype=np.float32)

  return blobs

def _get_image_blob(roidb, scale_inds):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)
  processed_ims = []
  im_scales = []
  for i in range(num_images):
    im = cv2.imread(roidb[i]['image'])
    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.MAX_SIZE)
    im_scales.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales

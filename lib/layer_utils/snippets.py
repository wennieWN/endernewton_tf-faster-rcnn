# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from layer_utils.generate_anchors import generate_anchors

def generate_anchors_pre(height, width, feat_stride, anchor_scales=(8,16,32), anchor_ratios=(0.5,1,2)):
  """ A wrapper function to generate anchors given different scales
    Also return the number of anchors in variable 'length'
  """
  anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
  A = anchors.shape[0]
  shift_x = np.arange(0, width) * feat_stride
  shift_y = np.arange(0, height) * feat_stride
  shift_x, shift_y = np.meshgrid(shift_x, shift_y)
  shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
  K = shifts.shape[0]
  # width changes faster, so here it is H, W, C
  anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
  anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
  length = np.int32(anchors.shape[0])

  return anchors, length

def generate_anchors_pre_tf(clp_filter, height, width, feat_stride=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):

  # old code
  # shift_x = tf.range(width) * feat_stride # width
  # shift_y = tf.range(height) * feat_stride # height
  # shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
  # sx = tf.reshape(shift_x, shape=(-1,))
  # sy = tf.reshape(shift_y, shape=(-1,))
  # shifts = tf.transpose(tf.stack([sx, sy, sx, sy]))
  # K = tf.multiply(width, height)
  #
  # shifts = tf.transpose(tf.reshape(shifts, shape=[1, K, 4]), perm=(1, 0, 2))
  #
  # anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
  # A = anchors.shape[0]
  # anchor_constant = tf.constant(anchors.reshape((1, A, 4)), dtype=tf.int32)
  #
  # length = K * A
  # anchors_tf = tf.reshape(tf.add(anchor_constant, shifts), shape=(length, 4))

  #todo: wn modified
  shift_x = tf.range(width) * feat_stride  # width
  shift_y = tf.range(height) * feat_stride  # height
  shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
  sx = tf.reshape(shift_x, shape=(-1,))
  sy = tf.reshape(shift_y, shape=(-1,))
  shifts = tf.transpose(tf.stack([sx, sy, sx, sy]))  # 竖着排
  K = tf.multiply(width, height)  # 原代码

  # 将点云图展开
  # clp_filter = tf.placeholder(tf.int32, shape=[None, None]) # 参数获取
  clp_filter_flat = tf.reshape(tf.transpose(clp_filter), (K, -1))  # shape: [23*32, 1]
  clp_filter_index = tf.where(clp_filter_flat)[:, 0]
  # 获取筛选后的坐标点
  shifts_filter = tf.gather(shifts, clp_filter_index)  # shpae: [none, 4]
  K = tf.size(clp_filter_index, out_type=tf.int32)
  shifts = tf.transpose(tf.reshape(shifts_filter, shape=[1, K, 4]), perm=(1, 0, 2))

  anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
  A = anchors.shape[0]
  anchor_constant = tf.constant(anchors.reshape((1, A, 4)), dtype=tf.int32)

  length = K * A
  anchors_tf = tf.reshape(tf.add(anchor_constant, shifts), shape=(length, 4))

  return tf.cast(anchors_tf, dtype=tf.float32), length

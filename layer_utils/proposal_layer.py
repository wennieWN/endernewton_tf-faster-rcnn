# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from model.config import cfg
from model.bbox_transform import bbox_transform_inv, clip_boxes, bbox_transform_inv_tf, clip_boxes_tf
from model.nms_wrapper import nms

def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
  """A simplified version compared to fast/er RCNN
     For details please see the technical report
  """
  if type(cfg_key) == bytes:
      cfg_key = cfg_key.decode('utf-8')
  pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
  post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

  # 0. 输入: rpn_cls_prob (1, ?, ?, 18)
  # 每个中心点的九个不同窗口的不含目标概率和含目标概率
  # 输入: rpn_bbox_pred (1, ?, ?, 36)

  # Get the scores and bounding boxes
  # 1. scores: (1, ?, ?, 9) =》 (?,)
  # 取rpn_cls_prob后9个: 每一个中心点的九个窗口的含目标概率
  scores = rpn_cls_prob[:, :, :, num_anchors:]
  scores = scores.reshape((-1, 1))

  # 2. rpn_bbox_pred: (?, 4)
  rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))

  # 3. 实际目标框的坐标proposals: (?, 4)
  proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
  proposals = clip_boxes(proposals, im_info[:2])

  # Pick the top region proposals
  # 4. 根据得分高低，取前pre_nms_topN（12000）个窗口
  # todo: 设定pre_nms_topN值
  order = scores.ravel().argsort()[::-1]
  if pre_nms_topN > 0:
    order = order[:pre_nms_topN]
  proposals = proposals[order, :]
  scores = scores[order]

  # Non-maximal suppression
  # 5. 非极大值抑制
  keep = nms(np.hstack((proposals, scores)), nms_thresh)

  # Pick th top region proposals after NMS
  # 6. 再选取前post_nms_topN（2000）个窗口
  # todo: 设定post_nms_topN值
  if post_nms_topN > 0:
    keep = keep[:post_nms_topN]
  proposals = proposals[keep, :]
  scores = scores[keep]

  # Only support single image as input
  # 7. 将每个窗口的信息由4维（坐标）扩至5维，包括窗口及其得分
  batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
  blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

  return blob, scores


def proposal_layer_tf(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
  if type(cfg_key) == bytes:
    cfg_key = cfg_key.decode('utf-8')
  pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
  post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

  # Get the scores and bounding boxes
  scores = rpn_cls_prob[:, :, :, num_anchors:]
  scores = tf.reshape(scores, shape=(-1,))
  rpn_bbox_pred = tf.reshape(rpn_bbox_pred, shape=(-1, 4))

  # 实际目标框的坐标 proposals: (?, 4)
  proposals = bbox_transform_inv_tf(anchors, rpn_bbox_pred)
  proposals = clip_boxes_tf(proposals, im_info[:2])

  # Non-maximal suppression
  # 取前post_nms_topN（2000）个窗口
  indices = tf.image.non_max_suppression(proposals, scores, max_output_size=post_nms_topN, iou_threshold=nms_thresh)

  # boxes (?, 4)
  # scores (?, 1)
  boxes = tf.gather(proposals, indices)
  boxes = tf.to_float(boxes)
  scores = tf.gather(scores, indices)
  scores = tf.reshape(scores, shape=(-1, 1))

  # Only support single image as input
  batch_inds = tf.zeros((tf.shape(indices)[0], 1), dtype=tf.float32)
  blob = tf.concat([batch_inds, boxes], 1)

  return blob, scores



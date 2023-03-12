# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import torch
import numpy as np
from tqdm import tqdm
import ocnn
#from thsolver import Solver, get_config
from solver import Solver, get_config
from datasets import (get_seg_shapenet_dataset, get_scannet_dataset,
                      get_kitti_dataset)

# The following line is to fix `RuntimeError: received 0 items of ancdata`.
# Refer: https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy('file_system')


class SegSolver(Solver):

  def get_model(self, flags):
    if flags.name.lower() == 'segnet':
      model = ocnn.models.SegNet(
          flags.channel, flags.nout, flags.stages, flags.interp, flags.nempty)
    elif flags.name.lower() == 'unet':
      model = ocnn.models.UNet(
          flags.channel, flags.nout, flags.interp, flags.nempty)
    else:
      raise ValueError
    return model

  def get_dataset(self, flags):
    if flags.name.lower() == 'shapenet':
      return get_seg_shapenet_dataset(flags)
    elif flags.name.lower() == 'scannet':
      return get_scannet_dataset(flags)
    elif flags.name.lower() == 'kitti':
      return get_kitti_dataset(flags)
    else:
      raise ValueError

  def get_input_feature(self, octree):
    flags = self.FLAGS.MODEL
    octree_feature = ocnn.modules.InputFeature(flags.feature, flags.nempty)
    data = octree_feature(octree)
    return data

  def model_forward(self, batch):
    octree = batch['octree'].cuda()
    points = batch['points'].cuda()
    data = self.get_input_feature(octree)
    query_pts = torch.cat([points.points, points.batch_id], dim=1)

    logit = self.model(data, octree, octree.depth, query_pts)
    label_mask = points.labels > self.FLAGS.LOSS.mask  # filter labels
    return logit[label_mask], points.labels[label_mask]

  def train_step(self, batch):
    logit, label = self.model_forward(batch)
    loss = self.loss_function(logit, label)
    return {'train/loss': loss}

  def test_step(self, batch):
    with torch.no_grad():
      logit, label = self.model_forward(batch)
    loss = self.loss_function(logit, label)
    accu = self.accuracy(logit, label)
    num_class = self.FLAGS.LOSS.num_class
    IoU, insc, union = self.IoU_per_shape(logit, label, num_class)

    names = ['test/loss', 'test/accu', 'test/mIoU'] + \
            ['test/intsc_%d' % i for i in range(num_class)] + \
            ['test/union_%d' % i for i in range(num_class)]
    tensors = [loss, accu, IoU] + insc + union
    return dict(zip(names, tensors))

  def eval_step(self, batch):
    with torch.no_grad():
      logit, _ = self.model_forward(batch)
    prob = torch.nn.functional.softmax(logit, dim=1)

    # The point cloud may be clipped when doing data augmentation. The
    # `inbox_mask` indicates which points are clipped. The `prob_all_pts`
    # contains the prediction for all points.
    inbox_mask = batch['inbox_mask'][0].cuda()
    assert len(batch['inbox_mask']) == 1, 'The batch_size must be 1'
    prob_all_pts = prob.new_zeros([inbox_mask.shape[0], prob.shape[1]])
    prob_all_pts[inbox_mask] = prob

    # Aggregate predictions across different epochs
    filename = batch['filename'][0]
    self.eval_rst[filename] = self.eval_rst.get(filename, 0) + prob_all_pts

    # Save the prediction results in the last epoch
    if self.FLAGS.SOLVER.eval_epoch - 1 == batch['epoch']:
      full_filename = os.path.join(self.logdir, filename + '.eval.npz')
      curr_folder = os.path.dirname(full_filename)
      if not os.path.exists(curr_folder): os.makedirs(curr_folder)
      np.savez(full_filename, prob=self.eval_rst[filename].cpu().numpy())

  def result_callback(self, avg_tracker, epoch):
    r''' Calculate the part mIoU for PartNet and ScanNet.
    '''

    iou_part = 0.0
    avg = avg_tracker.average()

    # Labels smaller than `mask` is ignored. The points with the label 0 in
    # PartNet are background points, i.e., unlabeled points
    mask = self.FLAGS.LOSS.mask + 1
    num_class = self.FLAGS.LOSS.num_class
    for i in range(mask, num_class):
      instc_i = avg['test/intsc_%d' % i]
      union_i = avg['test/union_%d' % i]
      iou_part += instc_i / (union_i + 1.0e-10)
    iou_part = iou_part / (num_class - mask)

    tqdm.write('=> Epoch: %d, test/mIoU_part: %f' % (epoch, iou_part))
    if self.summary_writer:
      self.summary_writer.add_scalar('test/mIoU_part', iou_part, epoch)

  def loss_function(self, logit, label):
    #criterion = torch.nn.CrossEntropyLoss()
    class_weights = [ 2.154,  1.445,  3.074,  1.686,  3.74 ,  3.504,  3.968,  3.966,
        3.896,  4.176,  4.887,  3.946,  5.852,  6.224,  5.606,  3.485,
        6.076,  3.965,  5.454,  4.229,  4.995,  5.126,  5.485,  5.841,
        5.604,  6.369,  4.682,  6.476,  5.713,  5.907,  6.069,  6.469,
        5.687,  6.472,  8.333,  5.304,  3.788,  6.075,  7.141,  7.411,
        8.169,  7.199,  6.231,  8.369,  7.028,  6.471,  4.97 ,  6.958,
        6.128,  5.055,  6.017,  6.032,  7.112,  6.322,  7.764,  7.058,
        8.451,  9.02 ,  6.973,  7.005,  6.631,  7.361,  6.377,  6.234,
        8.721,  6.232,  6.444,  7.534,  8.187,  6.125,  8.636,  7.569,
        7.495,  8.583,  6.809,  8.068, 10.339,  6.122,  7.79 ,  7.174,
        8.761,  5.647,  6.633,  8.729,  6.254,  9.123,  8.652,  8.971,
        9.586,  8.354,  8.027,  7.986,  8.239,  7.806,  8.998, 10.526,
        8.136,  9.906,  7.551,  8.577,  8.004,  5.915,  5.684,  8.851,
        7.439,  8.306,  8.231,  7.932,  9.693,  8.588,  8.58 ,  7.306,
        9.103, 10.767,  7.402,  9.898,  7.783, 10.518,  6.159,  5.224,
        8.988,  7.115,  9.751,  8.935, 10.384,  9.966,  8.341,  9.241,
        9.232,  8.157,  7.606,  8.455, 10.122,  9.949,  9.85 ,  7.866,
        7.236,  8.466, 10.34 ,  9.443, 11.175,  9.527, 11.908,  7.885,
        9.342,  9.848,  6.884,  9.297, 10.513, 10.252,  9.463, 11.423,
        8.416, 10.159,  9.055,  8.353, 10.322, 11.841, 10.421,  9.796,
        9.917,  9.728, 10.413, 10.208,  6.39 , 10.45 ,  9.123, 11.612,
        9.179,  7.773,  8.391, 10.293,  7.976,  5.691,  6.359,  6.804,
        8.088,  7.278,  8.589,  8.161,  9.662,  7.254,  8.98 ,  7.005,
       10.208,  9.305, 10.68 ,  7.988, 11.899, 10.016,  9.591,  9.829,
        8.324,  8.517, 11.676, 11.62 , 10.455,  8.805,  9.01 ,  9.621,
        8.639]
    criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to('cuda:0'))
    loss = criterion(logit, label.long())
    return loss

  def accuracy(self, logit, label):
    pred = logit.argmax(dim=1)
    accu = pred.eq(label).float().mean()
    return accu

  def IoU_per_shape(self, logit, label, class_num):
    pred = logit.argmax(dim=1)

    IoU, valid_part_num, esp = 0.0, 0.0, 1.0e-10
    intsc, union = [None] * class_num, [None] * class_num
    for k in range(class_num):
      pk, lk = pred.eq(k), label.eq(k)
      intsc[k] = torch.sum(torch.logical_and(pk, lk).float())
      union[k] = torch.sum(torch.logical_or(pk, lk).float())

      valid = torch.sum(lk.any()) > 0
      valid_part_num += valid.item()
      IoU += valid * intsc[k] / (union[k] + esp)

    # Calculate the shape IoU for ShapeNet
    IoU /= valid_part_num + esp
    return IoU, intsc, union

  @classmethod
  def update_configs(cls):
    FLAGS = get_config()
    FLAGS.LOSS.mask = -1           # mask the invalid labels


if __name__ == "__main__":
  SegSolver.main()

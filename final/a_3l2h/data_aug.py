from random import random, sample, uniform, choice
from itertools import product
from pathlib import Path
from typing import List, Optional, Tuple, Union
from copy import deepcopy
import logging
import os

import MinkowskiEngine as ME
import numpy as np
import albumentations as A
import scipy
import volumentations as V
import yaml

import torch
from torch.utils.data import Dataset

from .utils import read_plyfile

# for pytorch collate function

class VoxelizeCollateMerge:
    def __init__(
        self,
        ignore_label=255,
        voxel_size=1,
        mode="test",
        scenes=2,
        small_crops=False, # crop data according to (x, y) coordinates
        very_small_crops=False,
        # batch_instance=False,
        make_one_pc_noise=False, 
        place_nearby=False,
        place_far=False,
        proba=1,
    ):
        self.mode = mode
        self.scenes = scenes
        self.small_crops = small_crops
        self.very_small_crops = very_small_crops
        self.ignore_label = ignore_label
        self.voxel_size = voxel_size
        # self.batch_instance = batch_instance
        self.make_one_pc_noise = make_one_pc_noise
        self.place_nearby = place_nearby #?
        self.place_far = place_far #?
        self.proba = proba

    def __call__(self, batch):
        if self.mode == 'train':
            # batch is a list 
            if (
                ("train" in self.mode)
                and (not self.make_one_pc_noise)
                and (self.proba > random())
            ):
                if self.small_crops or self.very_small_crops:
                    batch = make_crops(batch)
                if self.very_small_crops:
                    batch = make_crops(batch)
                # if self.batch_instance:
                #     batch = batch_instances(batch)

                new_batch = []

                # batch: [scene0-[coord, feat, label-instance], scene1-[coord, feat, label-instance]]
                # len(batch) = nb 
                # self.scenes = 2 (number of scenes to be mixed)
                # combine two scenes into one scene
                for i in range(0, len(batch), self.scenes):
                    batch_coordinates = []
                    batch_features = []
                    batch_labels = []
                    for j in range(min(len(batch[i:]), self.scenes)):
                        batch_coordinates.append(batch[i + j][0])
                        batch_features.append(batch[i + j][1])
                        batch_labels.append(batch[i + j][2])
                    if (len(batch_coordinates) == 2) and self.place_nearby:
                        border = batch_coordinates[0][:, 0].max()
                        border -= batch_coordinates[1][:, 0].min()
                        batch_coordinates[1][:, 0] += border
                    elif (len(batch_coordinates) == 2) and self.place_far:
                        batch_coordinates[1] += (
                            np.random.uniform((-10, -10, -10), (10, 10, 10)) * 200
                        )
                    new_batch.append(
                        (
                            np.vstack(batch_coordinates),
                            np.vstack(batch_features),
                            np.concatenate(batch_labels),
                        )
                    )
                batch = new_batch

            # combine two scenes into one scene
            # create two data point with same (xyz RGB) but labels contain noise and labels from one scene
            # i.e. (xyz RGB, noise + label from scene1) and (xyz RGB, noise + label from scene2) 
            elif ("train" in self.mode) and self.make_one_pc_noise:
                new_batch = []
                for i in range(0, len(batch), 2):
                    # same logic but just different syntax 
                    if (i + 1) < len(batch):
                        new_batch.append(
                            [
                                np.vstack((batch[i][0], batch[i + 1][0])),
                                np.vstack((batch[i][1], batch[i + 1][1])),
                                np.concatenate(
                                    (
                                        batch[i][2],
                                        # add 
                                        np.full_like(batch[i + 1][2], self.ignore_label),
                                    )
                                ),
                            ]
                        )
                        new_batch.append(
                            [
                                np.vstack((batch[i][0], batch[i + 1][0])),
                                np.vstack((batch[i][1], batch[i + 1][1])),
                                np.concatenate(
                                    (
                                        np.full_like(batch[i][2], self.ignore_label),
                                        batch[i + 1][2],
                                    )
                                ),
                            ]
                        )
                    else:
                        new_batch.append([batch[i][0], batch[i][1], batch[i][2]])
                batch = new_batch
            
            voxelized_data = voxelize(batch, self.ignore_label, self.voxel_size)
        
        elif self.mode == 'valid':
            voxelized_data = voxelize(batch, self.ignore_label, self.voxel_size)
        
        elif self.mode == 'test':
            voxelized_data = voxelize_test(batch, self.ignore_label, self.voxel_size)
        return voxelized_data

class PointCloudCollateMerge:
    def __init__(
        self,
        ignore_label=255,
        mode="test",
        scenes=2,
        small_crops=False,
        very_small_crops=False,
        # batch_instance=False,
        make_one_pc_noise=False,
        place_nearby=False,
        place_far=False,
        proba=1,
    ):
        self.mode = mode
        self.scenes = scenes
        self.small_crops = small_crops
        self.very_small_crops = very_small_crops
        self.ignore_label = ignore_label
        # self.batch_instance = batch_instance
        self.make_one_pc_noise = make_one_pc_noise
        self.place_nearby = place_nearby
        self.place_far = place_far
        self.proba = proba

    def merge_to_batch(self, batch):
        coords, feats, = [], []
        for sample in batch:
            coord, feat = sample[0], sample[1]
            coords.append(coord)
            feats.append(feat)

        coords = np.vstack(coords)
        feats = np.vstack(feats)

        if self.mode == 'test':
            return coords, feats
        else:
            labels = []
            for sample in batch:
                label = sample[2]
                labels.append(label)
            labels = np.concatenate(labels)
            return coords, feats, labels

    def __call__(self, batch):
        # batch is a list
        if self.mode == 'train':
            if (
                ("train" in self.mode)
                and (not self.make_one_pc_noise)
                and (self.proba > random())
            ):
                if self.small_crops or self.very_small_crops:
                    batch = make_crops(batch)
                if self.very_small_crops:
                    batch = make_crops(batch)
                # if self.batch_instance:
                #     batch = batch_instances(batch)
                new_batch = []

                # batch: [scene0-[coord, feat, label-instance], scene1-[coord, feat, label-instance]]
                # len(batch) = nb 
                # self.scenes = 2 (number of scenes to be mixed)
                # combine two scenes into one scene
                # number of new batch = number of original batch / 2
                for i in range(0, len(batch), self.scenes):
                    batch_coordinates = []
                    batch_features = []
                    batch_labels = []
                    for j in range(min(len(batch[i:]), self.scenes)):
                        batch_coordinates.append(batch[i + j][0])
                        batch_features.append(batch[i + j][1])
                        batch_labels.append(batch[i + j][2])

                    if (len(batch_coordinates) == 2) and self.place_nearby:
                        border = batch_coordinates[0][:, 0].max()
                        border -= batch_coordinates[1][:, 0].min()
                        batch_coordinates[1][:, 0] += border
                    elif (len(batch_coordinates) == 2) and self.place_far:
                        batch_coordinates[1] += (
                            np.random.uniform((-10, -10, -10), (10, 10, 10)) * 200
                        )
                    new_batch.append(
                        (
                            np.vstack(batch_coordinates),
                            np.vstack(batch_features),
                            np.concatenate(batch_labels),
                        )
                    )
                batch = new_batch

            # combine two scenes into one scene
            # create two data point with same (xyz RGB) but labels contain noise and labels from one scene
            # i.e. (xyz RGB, noise + label from scene1) and (xyz RGB, noise + label from scene2) 
            elif ("train" in self.mode) and self.make_one_pc_noise:
                new_batch = []
                for i in range(0, len(batch), 2):
                    # same logic but just different syntax 
                    if (i + 1) < len(batch):
                        new_batch.append(
                            [
                                np.vstack((batch[i][0], batch[i + 1][0])),
                                np.vstack((batch[i][1], batch[i + 1][1])),
                                np.concatenate(
                                    (
                                        batch[i][2],
                                        # add 
                                        np.full_like(batch[i + 1][2], self.ignore_label),
                                    )
                                ),
                            ]
                        )
                        new_batch.append(
                            [
                                np.vstack((batch[i][0], batch[i + 1][0])),
                                np.vstack((batch[i][1], batch[i + 1][1])),
                                np.concatenate(
                                    (
                                        np.full_like(batch[i][2], self.ignore_label),
                                        batch[i + 1][2],
                                    )
                                ),
                            ]
                        )
                    else:
                        new_batch.append([batch[i][0], batch[i][1], batch[i][2]])
                batch = new_batch

            coordinates, features, labels = self.merge_to_batch(batch)
            
            data = {
                'coordinates': torch.from_numpy(coordinates).float(),
                'features': torch.from_numpy(features).float(),
                'labels': torch.from_numpy(labels).long(),
            }

        elif self.mode == 'valid':
            coordinates, features, labels = self.merge_to_batch(batch)
            
            data = {
                'coordinates': torch.from_numpy(coordinates).float(),
                'features': torch.from_numpy(features).float(),
                'labels': torch.from_numpy(labels).long(),
            }

        elif self.mode == 'test':
            coordinates, features = self.merge_to_batch(batch)
            data = {
                'coordinates': torch.from_numpy(coordinates).float(),
                'features': torch.from_numpy(features).float(),       
            }
        return data

# create data point that only contains (xyzRGB, label of an instance)
def batch_instances(batch):
    new_batch = []
    for sample in batch:
        for instance_id in np.unique(sample[2][:, 1]):
            new_batch.append(
                (
                    sample[0][sample[2][:, 1] == instance_id],
                    sample[1][sample[2][:, 1] == instance_id],
                    sample[2][sample[2][:, 1] == instance_id][:, 0],
                ),
            )
    return new_batch

def voxelize_test(batch, ignore_label, voxel_size):
    (coordinates, features, inverse_maps) = (
        [],
        [],
        []
    )

    voxelization_dict = {
        "ignore_label": ignore_label,
        # "quantization_size": self.voxel_size,
        "return_index": True,
        "return_inverse": True,
        "return_maps_only": True,
    }

    # batch: [mixed scene0-[coord, feat, label-instance], mixed scene1-[coord, feat, label-instance]]
    for sample in batch:

        coords = np.floor(sample[0] / voxel_size)
        # print(coords.shape) (# of point clouds, 3)
        voxelization_dict.update(
                {"coordinates": np.ascontiguousarray(coords), 
                "features": np.ascontiguousarray(sample[1])
                })
        unique_map, inverse_map = \
                ME.utils.sparse_quantize(**voxelization_dict)
        
        inverse_maps.append(inverse_map)

        sample_coordinates = coords[unique_map]
        coordinates.append(torch.from_numpy(sample_coordinates).int())
        sample_features = sample[1][unique_map]
        features.append(torch.from_numpy(sample_features).float())

    # Concatenate all lists
    input_dict = {"coords": coordinates, "feats": features}
    coordinates, features = ME.utils.sparse_collate(**input_dict)
    labels_voxelized = torch.Tensor([])

    # return ME output
    # (nb, (coord, feat, labels, ...))
    data = {   
        'coordinates': coordinates,
        'features': features,
        'inverse_maps': inverse_maps
    }
    return data

def voxelize(batch, ignore_label, voxel_size):
    (coordinates, features, labels_voxelized, labels_original, inverse_maps,) = (
        [],
        [],
        [],
        [],
        [],
    )

    voxelization_dict = {
        "ignore_label": ignore_label,
        # "quantization_size": self.voxel_size,
        "return_index": True,
        "return_inverse": True,
        "return_maps_only": True,
    }

    # batch: [mixed scene0-[coord, feat, label-instance], mixed scene1-[coord, feat, label-instance]]
    for sample in batch:
        labels_original.append(sample[2])

        coords = np.floor(sample[0] / voxel_size)
        # print(coords.shape) (# of point clouds, 3)
        voxelization_dict.update(
                {"coordinates": np.ascontiguousarray(coords), 
                "features": np.ascontiguousarray(sample[1])
                })
        unique_map, inverse_map = \
                ME.utils.sparse_quantize(**voxelization_dict)
        
        inverse_maps.append(inverse_map)

        sample_coordinates = coords[unique_map]
        coordinates.append(torch.from_numpy(sample_coordinates).int())
        sample_features = sample[1][unique_map]
        features.append(torch.from_numpy(sample_features).float())
        if len(sample[2]) > 0:
            sample_labels = sample[2][unique_map]
            labels_voxelized.append(torch.from_numpy(sample_labels).long())

    # Concatenate all lists
    input_dict = {"coords": coordinates, "feats": features}
    if len(labels_voxelized) > 0:
        input_dict["labels"] = labels_voxelized
        coordinates, features, labels_voxelized = \
            ME.utils.sparse_collate(**input_dict)
    else:
        coordinates, features = ME.utils.sparse_collate(**input_dict)
        labels_voxelized = torch.Tensor([])
    
    # return ME output
    # (nb, (coord, feat, labels, ...))
    data = {   
        'coordinates': coordinates,
        'features': features,
        'labels_voxelized': labels_voxelized,
        'labels_original': labels_original,
        'inverse_maps': inverse_maps
    }
    return data

def make_crops(batch):
    new_batch = []
    # detupling
    for scene in batch:
        # [scene0-[coord, feat, label], scene1-[coord, feat, label]]
        new_batch.append([scene[0], scene[1], scene[2]])
    batch = new_batch
    new_batch = []
    for scene in batch:
        # move to center for better quadrant split
        scene[0][:, :3] -= scene[0][:, :3].mean(0)

        # quadrant: each of four quarters of a circle.
        # BUGFIX - there always would be a point in every quadrant
        scene[0] = np.vstack(
            (
                scene[0],
                np.array(
                    [
                        [0.1, 0.1, 0.1],
                        [0.1, -0.1, 0.1],
                        [-0.1, 0.1, 0.1],
                        [-0.1, -0.1, 0.1],
                    ]
                ),
            )
        )

        scene[1] = np.vstack((scene[1], np.zeros((4, scene[1].shape[1]))))
        # label: 255 (ignore label)
        scene[2] = np.concatenate((scene[2], np.full_like((scene[2]), 255)[:4]))

        # crop smaller regions of a scene
        crop = scene[0][:, 0] > 0
        crop &= scene[0][:, 1] > 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

        crop = scene[0][:, 0] > 0
        crop &= scene[0][:, 1] < 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

        crop = scene[0][:, 0] < 0
        crop &= scene[0][:, 1] > 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

        crop = scene[0][:, 0] < 0
        crop &= scene[0][:, 1] < 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

    # moving all of them to center
    for i in range(len(new_batch)):
        new_batch[i][0][:, :3] -= new_batch[i][0][:, :3].mean(0)
    return new_batch

class SemanticSegmentationDataset(Dataset):
    """Docstring for SemanticSegmentationDataset. """

    def __init__(
        self,
        split_path = None, # train_split.txt or # val_split.txt 
        color_mean_std_path = None, # train_color_mean_std.npy or val_color_mean_std.npy
        data_dir: Optional[Union[str, Tuple[str]]] = "./dataset",

        # mean std values from scannet
        color_mean_std: Optional[Union[str, Tuple[Tuple[float]]]] = (
            (0.47793125906962, 0.4303257521323044, 0.3749598901421883),
            (0.2834475483823543, 0.27566157565723015, 0.27018971370874995),
        ),
        mode: Optional[str] = "train",
        ignore_label: Optional[Union[int, Tuple[int]]] = 255,
        # instance_oversampling=0,
        flip_in_center=False,
        add_color=False,
    ):

        self.mode = mode
        self.split_path = split_path 
        self.data_dir = data_dir

        self.ignore_label = ignore_label
        self.flip_in_center = flip_in_center

        # preprocess txt
        if self.mode != 'test':
            self.data = []
            with open(self.split_path) as f:
                contents = f.readlines()
                for content in contents:
                    self.data.append(content.strip())
        elif self.mode == 'test':
            self.data_dir = os.path.join(self.data_dir, 'test')
            self.data = os.listdir(self.data_dir)

        self.add_colors = add_color

        # normalize color channels
        if color_mean_std_path is not None:
            with open(color_mean_std_path, 'rb') as f:
                color_mean = np.load(f)
                color_std = np.load(f)

        elif len(color_mean_std[0]) == 3 and len(color_mean_std[1]) == 3:
            color_mean, color_std = color_mean_std[0], color_mean_std[1]

        # volume augmentations for (x, y, z)
        self.volume_augmentations = V.Compose([
            V.augmentations.transforms.Scale3d(always_apply=True),
            V.augmentations.transforms.RotateAroundAxis3d(always_apply=True, axis=(0,0,1), rotation_limit=[-np.pi, np.pi]),
            V.augmentations.transforms.RotateAroundAxis3d(always_apply=True, axis=(0,1,0), rotation_limit=[-np.pi/24, np.pi/24]),
            V.augmentations.transforms.RotateAroundAxis3d(always_apply=True, axis=(1,0,0), rotation_limit=[-np.pi/24, np.pi/24])
            ])
        
        # additionally apply random brightness and contrast aug- mentation as well as color-jitter
        self.image_augmentations = A.Compose([
            A.augmentations.transforms.RandomBrightness(always_apply=True),
            A.augmentations.transforms.RandomContrast(always_apply=True),
            A.augmentations.transforms.ColorJitter(always_apply=True)
        ])

        # mandatory color augmentation
        if add_color:
            self.normalize_color = A.Normalize(
                mean=color_mean, std=color_std
                )


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        point_path = os.path.join(self.data_dir, self.data[idx])
        points = read_plyfile(point_path)
        
        if self.mode != 'test':
            coordinates, color, labels, instances = (
                points[:, :3],
                points[:, 3:6],
                points[:, 6],
                points[:, 7],
            )
        
        else:
            coordinates, color = (
                points[:, :3],
                points[:, 3:6]
            )

        # volume and image augmentations for train
        if "train" in self.mode:
            # subtract the centroid from all point positions
            coordinates -= coordinates.mean(0)
            coordinates += np.random.uniform(coordinates.min(0), coordinates.max(0)) / 2

            # flip point cloud
            if self.flip_in_center:
                coordinates = flip_in_center(coordinates)

            for i in (0, 1):
                if random() < 0.5:
                    coord_max = np.max(points[:, i])
                    coordinates[:, i] = coord_max - coordinates[:, i]

            # elastic distortion
            if random() < 0.95:
                for granularity, magnitude in ((0.2, 0.4), (0.8, 1.6)):
                    coordinates = elastic_distortion(
                        coordinates, granularity, magnitude
                    )

            # randomly scale the scene by Unif [0.9, 1.1]
            # randomly rotate the scene along the up-right axis
            aug = self.volume_augmentations(
                points=coordinates, features=color, labels=labels,
            )

            coordinates, color, labels = (
                aug["points"],
                aug["features"],
                aug["labels"],
            )

            
            pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
            color = np.squeeze(self.image_augmentations(image=pseudo_image)["image"])

            ####### finish data augmentation #####

        # normalize color information
        pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
        color = np.squeeze(self.normalize_color(image=pseudo_image)["image"])

        # # prepare labels and map from 0 to 20(40)
        # labels = labels.astype(np.int32)
        # if labels.size > 0:
        #     labels[:, 0] = self._remap_from_zero(labels[:, 0])
        #     if not self.add_instance:
        #         # taking only first column, which is segmentation label, not instance
        #         labels = labels[:, 0].flatten()

        # (x, y, z), (r, g, b), (label)
        if self.mode != 'test':
            return coordinates, color, labels
        else:
            return coordinates, color





    # add instance into individual scenes
    # need instance database (but don't know how to get it)

    # def augment_individual_instance(
    #     self, coordinates, color, labels, oversampling=1.0
    # ):
    #     max_instance = int(len(np.unique(labels[:, 1])))
    #     # randomly selecting half of non-zero instances
    #     for instance in range(0, int(max_instance * oversampling)):
    #         if self.place_around_existing:
    #             center = choice(
    #                 coordinates[labels[:, 1] == choice(np.unique(labels[:, 1]))]
    #             )
    #         else:
    #             center = np.array([uniform(-5, 5), uniform(-5, 5), uniform(-0.5, 2)])
    #         instance = choice(choice(self.instance_data))
    #         instance = np.load(instance["instance_filepath"])
    #         # centering two objects
    #         instance[:, :3] = instance[:, :3] - instance[:, :3].mean(axis=0) + center
    #         max_instance = max_instance + 1
    #         instance[:, -1] = max_instance
    #         aug = V.Compose(
    #             [
    #                 V.Scale3d(),
    #                 V.RotateAroundAxis3d(rotation_limit=np.pi / 24, axis=(1, 0, 0)),
    #                 V.RotateAroundAxis3d(rotation_limit=np.pi / 24, axis=(0, 1, 0)),
    #                 V.RotateAroundAxis3d(rotation_limit=np.pi, axis=(0, 0, 1)),
    #             ]
    #         )(
    #             points=instance[:, :3],
    #             features=instance[:, 3:6],
    #             labels=instance[:, 9:],
    #         )
    #         coordinates = np.concatenate((coordinates, aug["points"]))
    #         color = np.concatenate((color, aug["features"]))
    #         labels = np.concatenate((labels, aug["labels"]))

    #     return coordinates, color, labels


def elastic_distortion(pointcloud, granularity, magnitude):
    """Apply elastic distortion on sparse coordinate space.

    pointcloud: numpy array of (number of points, at least 3 spatial dims)
    granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
    magnitude: noise multiplier
  """
    blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
    blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
    blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
    coords = pointcloud[:, :3]
    coords_min = coords.min(0)

    # Create Gaussian noise tensor of the size given by granularity.
    noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
    noise = np.random.randn(*noise_dim, 3).astype(np.float32)

    # Smoothing.
    for _ in range(2):
        noise = scipy.ndimage.filters.convolve(noise, blurx, mode="constant", cval=0)
        noise = scipy.ndimage.filters.convolve(noise, blury, mode="constant", cval=0)
        noise = scipy.ndimage.filters.convolve(noise, blurz, mode="constant", cval=0)

    # Trilinear interpolate noise filters for each spatial dimensions.
    ax = [
        np.linspace(d_min, d_max, d)
        for d_min, d_max, d in zip(
            coords_min - granularity,
            coords_min + granularity * (noise_dim - 2),
            noise_dim,
        )
    ]
    interp = scipy.interpolate.RegularGridInterpolator(
        ax, noise, bounds_error=0, fill_value=0
    )
    pointcloud[:, :3] = coords + interp(coords) * magnitude
    return pointcloud


def flip_in_center(coordinates):
    # moving coordinates to center
    coordinates -= coordinates.mean(0)
    aug = V.Compose(
        [
            V.Flip3d(axis=(0, 1, 0), always_apply=True),
            V.Flip3d(axis=(1, 0, 0), always_apply=True),
        ]
    )

    first_crop = coordinates[:, 0] > 0
    first_crop &= coordinates[:, 1] > 0
    # x -y
    second_crop = coordinates[:, 0] > 0
    second_crop &= coordinates[:, 1] < 0
    # -x y
    third_crop = coordinates[:, 0] < 0
    third_crop &= coordinates[:, 1] > 0
    # -x -y
    fourth_crop = coordinates[:, 0] < 0
    fourth_crop &= coordinates[:, 1] < 0

    if first_crop.size > 1:
        coordinates[first_crop] = aug(points=coordinates[first_crop])["points"]
    if second_crop.size > 1:
        minimum = coordinates[second_crop].min(0)
        minimum[2] = 0
        minimum[0] = 0
        coordinates[second_crop] = aug(points=coordinates[second_crop])["points"]
        coordinates[second_crop] += minimum
    if third_crop.size > 1:
        minimum = coordinates[third_crop].min(0)
        minimum[2] = 0
        minimum[1] = 0
        coordinates[third_crop] = aug(points=coordinates[third_crop])["points"]
        coordinates[third_crop] += minimum
    if fourth_crop.size > 1:
        minimum = coordinates[fourth_crop].min(0)
        minimum[2] = 0
        coordinates[fourth_crop] = aug(points=coordinates[fourth_crop])["points"]
        coordinates[fourth_crop] += minimum

    return coordinates




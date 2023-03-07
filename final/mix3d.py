from a_3l2h.data_aug import SemanticSegmentationDataset, VoxelizeCollateMerge, PointCloudCollateMerge
from torch.utils.data import DataLoader

def main():
    train_ds = SemanticSegmentationDataset(
        split_path='train_split.txt',
        color_mean_std_path='train_color_mean_std.npy',
        mode='train',
        ignore_label=255,
        flip_in_center=True,
        add_color=True
    )


    val_ds = SemanticSegmentationDataset(
        split_path='val_split.txt',
        color_mean_std_path='train_color_mean_std.npy', # use train color mean and std to normalize 
        mode='val',
        ignore_label=255,
        flip_in_center=True,
        add_color=True
    )
    
    test_ds = SemanticSegmentationDataset(
        color_mean_std_path='train_color_mean_std.npy', # use train color mean and std to normalize 
        mode='test',
        ignore_label=255,
        flip_in_center=True,
        add_color=True
    )

    ################## Voxelized data loader ###########

    voxelize_collate_train = VoxelizeCollateMerge(
        ignore_label=255,
        voxel_size=1,
        mode='train',
        scenes=2, # number of scenes to be combines
        small_crops=False, # crop data according to (x, y) coordinates, not sure what it is for?
        very_small_crops=False, # crop again
        make_one_pc_noise=False, 
        place_nearby=False, # ?
        place_far=False, # ?
    )
        # make_one_pc_noise=True
        # combine two scenes into one scene
        # create two data point with same (xyz RGB) but labels contain noise and labels from one scene
        # i.e. (xyz RGB, noise + label from scene1) and (xyz RGB, noise + label from scene2) 

    vox_train_dl = DataLoader(
        train_ds,
        batch_size=10,
        shuffle=True,
        collate_fn=voxelize_collate_train
    )

    # d = next(iter(vox_train_dl))

    # print("="*50)
    # print(
    #     d['coordinates'].size(), # (number of total voxel, 4), tensor
    #     d['features'].size(), # (number of total voxel, 3), tensor
    #     len(d['labels_voxelized']), # (number of total voxel), tensor
    #     len(d['labels_original']), # (batch size, )
    #     len(d['inverse_maps'])) # (batch size, )
    
    # print(d['labels_original']) # list of array
    # print(d['inverse_maps']) # list of array

    voxelize_collate_val = VoxelizeCollateMerge(
        ignore_label=255,
        voxel_size=1,
        mode='valid'
    )
    voxelize_collate_test = VoxelizeCollateMerge(
        ignore_label=255,
        voxel_size=1,
        mode='test'
    )

    vox_val_dl = DataLoader(
        val_ds,
        batch_size=10,
        shuffle=False,
        collate_fn=voxelize_collate_val
    )

    vox_test_dl = DataLoader(
        test_ds,
        batch_size=10,
        shuffle=False,
        collate_fn=voxelize_collate_test
    )

    # d = next(iter(vox_val_dl))
    # print(d)

    # d = next(iter(vox_test_dl))
    # print(d['coordinates'], d['features'], d['inverse_maps'])


    ############### Point cloud dataloader ###############
    pointcloud_collate_train = PointCloudCollateMerge(
        ignore_label=255,
        mode='train',
        scenes=2, # number of scenes to be combines
        small_crops=False, # crop data according to (x, y) coordinates, not sure what it is for?
        very_small_crops=False, # crop again
        make_one_pc_noise=False, 
        place_nearby=False, # ?
        place_far=False, # ?
    )

    pc_train_dl = DataLoader(
        train_ds,
        batch_size=10,
        shuffle=True,
        collate_fn=pointcloud_collate_train
    )

    # d = next(iter(pc_train_dl))

    # print("="*50)
    # print(
    #     d['coordinates'].size(), # (number of total data points, 4), tensor
    #     d['features'].size(), # (number of total data points, 3), tensor
    #     d['labels'].size(), # (number of total data points, )
    # )

    pointcloud_collate_val = PointCloudCollateMerge(
        ignore_label=255,
        mode='valid',
    )

    pc_val_dl = DataLoader(
        val_ds,
        batch_size=10,
        shuffle=False,
        collate_fn=pointcloud_collate_val
    )

    # d = next(iter(pc_val_dl))
    # print(d['coordinates'], d['features'], d['labels'])

    pointcloud_collate_test = PointCloudCollateMerge(
        ignore_label=255,
        mode='test',
    )

    pc_test_dl = DataLoader(
        test_ds,
        batch_size=10,
        shuffle=False,
        collate_fn=pointcloud_collate_test
    )

    # d = next(iter(pc_test_dl))
    # print(d['coordinates'], d['features'])

if __name__ == '__main__':
    main()
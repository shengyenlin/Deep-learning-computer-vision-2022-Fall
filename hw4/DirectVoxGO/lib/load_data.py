import numpy as np

from .load_blender import load_blender_data, load_blender_data_no_gt


def load_data(cfg, args):

    K, depths = None, None
    near_clip = None

    if cfg.dataset_type == 'blender':
        if args.render_test_no_gt:
            imgs_name, poses, render_poses, hwf = load_blender_data_no_gt(args.test_json_path, cfg.half_res, cfg.testskip)
        else:
            imgs_name, images, poses, render_poses, hwf, i_split = load_blender_data(cfg.datadir, cfg.half_res, cfg.testskip)
            print('Loaded blender', images.shape, render_poses.shape, hwf, cfg.datadir)
            i_train, i_val, i_test = i_split

            if images.shape[-1] == 4:
                if cfg.white_bkgd:
                    images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
                else:
                    images = images[...,:3]*images[...,-1:]
                    
        near, far = 2., 6.

    else:
        raise NotImplementedError(f'Unknown dataset type {cfg.dataset_type} exiting')

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if args.render_test_no_gt:
        # (50, 2) with each data point = (800, 800)
        HW = [np.array([H, W]) for i in range(len(imgs_name))]
        HW = np.stack(HW, axis=0)
    else:
        # (150, 2) with each data point = (800, 800)
        HW = np.array([im.shape[:2] for im in images]) 
        irregular_shape = (images.dtype is np.dtype('object'))

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if len(K.shape) == 2:
        Ks = K[None].repeat(len(poses), axis=0)
    else:
        Ks = K

    render_poses = render_poses[...,:4]

    if args.render_test_no_gt:
        data_dict = dict(
            hwf=hwf, HW=HW, Ks=Ks,
            near=near, far=far, near_clip=near_clip,
            poses=poses, render_poses=render_poses,
            depths=depths, imgs_name=imgs_name
        )
    else:
        data_dict = dict(
            hwf=hwf, HW=HW, Ks=Ks,
            near=near, far=far, near_clip=near_clip,
            i_train=i_train, i_val=i_val, i_test=i_test,
            poses=poses, render_poses=render_poses,
            images=images, depths=depths,
            irregular_shape=irregular_shape,
            imgs_name=imgs_name
        )
    return data_dict


def inward_nearfar_heuristic(cam_o, ratio=0.05):
    dist = np.linalg.norm(cam_o[:,None] - cam_o, axis=-1)
    far = dist.max()  # could be too small to exist the scene bbox
                      # it is only used to determined scene bbox
                      # lib/dvgo use 1e9 as far
    near = far * ratio
    return near, far


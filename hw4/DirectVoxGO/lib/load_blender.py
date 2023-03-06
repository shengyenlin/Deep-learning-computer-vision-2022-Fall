import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    # splits = ['train', 'val', 'test]
    splits = ['train', 'val', 'val']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)
    
    all_imgs_name = []
    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        imgs_name = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            file_name = os.path.join(frame['file_path'] + '.png')
            fname = os.path.join(basedir, file_name)
            imgs_name.append(file_name)
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs_name.append(imgs_name)
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    # [
    # array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
    #        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    #        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]), 
    # array([50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
    #        67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
    #        84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]), 
    # array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
    #        113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
    #        126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138,
    #        139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149])
    # ]
    print(i_split)

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,160+1)[:-1]], 0)

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return all_imgs_name, imgs, poses, render_poses, [H, W, focal], i_split

def load_blender_data_no_gt(test_json_path, half_res=False, testskip=1):
    with open(test_json_path, 'r') as fp:
        test_data = json.load(fp)
    
    imgs_name = []
    poses = []
    #print(test_data['frames'])
    for frame in test_data['frames']:
        file_name = frame['file_path'].split("/")[-1]
        imgs_name.append(file_name)
        poses.append(np.array(frame['transform_matrix']))

    poses = np.array(poses).astype(np.float32)

    # H, W = imgs[0].shape[:2]
    H, W = 800, 800
    camera_angle_x = float(test_data['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack(
        [pose_spherical(angle, -30.0, 4.0) \
            for angle in np.linspace(-180,180,160+1)[:-1]], 0
            )

    return imgs_name, poses, render_poses, [H, W, focal]



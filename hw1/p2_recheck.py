import os
import sys

import numpy as np
import imageio.v2 as imageio

GT_PATH = sys.argv[1]
PRED_PATH = sys.argv[2]

def read_mask(path):
    mask = imageio.imread(path) #512x512x3
    #not np.empty
    masks = np.zeros((mask.shape[0],mask.shape[1]))

    #load mask
    mask = (mask >= 128).astype(int)
    #512x512x3 -> 512x512
    mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
    masks[mask == 3] = 0  # (Cyan: 011) Urban land 
    masks[mask == 6] = 1  # (Yellow: 110) Agriculture land 
    masks[mask == 5] = 2  # (Purple: 101) Rangeland 
    masks[mask == 2] = 3  # (Green: 010) Forest land 
    masks[mask == 1] = 4  # (Blue: 001) Water 
    masks[mask == 7] = 5  # (White: 111) Barren land 
    masks[mask == 0] = 6  # (Black: 000) Unknown 
    return masks

def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6

    return mean_iou

def main():
    pred_img_paths = os.listdir(PRED_PATH)
    pred_img_paths = sorted(pred_img_paths)

    gt_img_path_tmp = os.listdir(GT_PATH)
    gt_img_path_tmp = sorted(gt_img_path_tmp)
    gt_img_path = []
    for img_path in gt_img_path_tmp:
        suffix = img_path.split('.')[1]
        if suffix =='png':
            gt_img_path.append(img_path)

    gt_masks = []
    for img_path in gt_img_path:
        mask = read_mask(
            os.path.join(GT_PATH, img_path)
            )
        gt_masks.append(mask)

    pred_masks = []
    for img_path in pred_img_paths:
        mask = read_mask(
            os.path.join(PRED_PATH, img_path)
            )
        pred_masks.append(mask)

    gt_masks = np.array(gt_masks)
    pred_masks = np.array(pred_masks)

    iou = mean_iou_score(pred_masks, gt_masks)
    print(f"Recheck iou: {iou}")

if __name__ == '__main__':
    main()
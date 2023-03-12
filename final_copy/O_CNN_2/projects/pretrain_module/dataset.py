from torch.utils import data
import torch
import ocnn
from thsolver import Dataset
import sys
sys.path.append('..')

from datasets.scannet import ScanNetTransform, ReadPly

class CollatDualeBatch:
    r''' Merge a list of octrees and points into a batch.
    '''

    def __init__(self, merge_points: bool = False):
        self.merge_points = merge_points

    def solve_one_batch(self, batch: list):
        assert type(batch) == list

        outputs = {}
        for key in batch[0].keys():
            outputs[key] = [b[key] for b in batch]

        # Merge a batch of octrees into one super octree
            if 'octree' in key:
                octree = ocnn.octree.merge_octrees(outputs[key])
                # NOTE: remember to construct the neighbor indices
                octree.construct_all_neigh()
                outputs[key] = octree

            # Merge a batch of points
            if 'points' in key and self.merge_points:
                outputs[key] = ocnn.octree.merge_points(outputs[key])

            # Convert the labels to a Tensor
            if 'label' in key:
                outputs['label'] = torch.LongTensor(outputs[key])

        return outputs

    def __call__(self, batch):
        batch1, batch2 = self.solve_one_batch(batch), self.solve_one_batch(batch)
        return batch1, batch2

    

def get_dual_pretraining_dataloader(flags):
    transform = ScanNetTransform(flags)
    read_ply = ReadPly(has_normal=True, has_color=True, has_label=True)
    collate_batch = CollatDualeBatch(merge_points=True)

    dataset = Dataset(flags.location, flags.filelist, transform,
                        read_file=read_ply)
    dl = data.DataLoader(
                dataset, 
                batch_size=flags.batch_size, 
                num_workers=flags.num_workers,
                            collate_fn=collate_batch, 
                            pin_memory=True
    )
    return dl

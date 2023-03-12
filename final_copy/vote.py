from argparse import ArgumentParser
import zipfile
import tempfile
from glob import glob
import os
from collections import Counter, OrderedDict
from typing import List

def read_zip(file: str):
    filenames, points = [], []
    with tempfile.TemporaryDirectory() as tempdir:
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(tempdir)
        for name in glob(os.path.join(tempdir, '*.txt')):
            if name.endswith('.txt'):
                with open(name) as f:
                    filenames.append(os.path.split(name)[-1]) 
                    points.append([int(line.strip()) for line in f.readlines()])
    return zip(*sorted(zip(filenames, points), key=lambda x: x[0]))

def write_zip(result, dest):
    with zipfile.ZipFile(dest, 'w') as zip_obj:
        with tempfile.TemporaryDirectory() as tempdir:
            for name, value in result.items():
                path = os.path.join(tempdir, name)
                with open(path, 'w', encoding='ascii')as f:
                    # print(value)
                    value = list(map(str, value))
                    f.write('\n'.join(value))
                    f.write('\n')
                zip_obj.write(path, arcname=name)


def batched_bincount(x, dim, max_value=1000):
    target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
    values = torch.ones_like(x)
    target.scatter_add_(dim, x, values)
    return target

class OrderedCounter(Counter, OrderedDict):
    pass


if __name__ == '__main__':
    import torch
    from collections import defaultdict
    parser = ArgumentParser()
    parser.add_argument('--inp', nargs='+', help='voting candidates *.zip')
    parser.add_argument('--dest', default='vote.zip')
    
    args = parser.parse_args()
    filenames = []
    points_list = []
    for inp in args.inp:
        fnames, points = read_zip(inp) # [fnames, lines]
        points_list.append(points)
    
    result = defaultdict(lambda: [])
    for i, fname in enumerate(fnames):
        print(fname)
        matrix = [points_list[j][i] for j in range(len(args.inp))] #[num_zip, point num]
        for j in range(len(matrix[0])):
            point = OrderedCounter([matrix[k][j] for k in range(len(args.inp))]).most_common(1)[0][0]
            
            result[fname].append(point)
            
    write_zip(result, args.dest)
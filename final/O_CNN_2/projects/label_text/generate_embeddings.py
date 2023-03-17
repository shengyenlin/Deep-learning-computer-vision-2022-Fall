from argparse import ArgumentParser
import json
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
from constant import CLASS_LABELS_200, VALID_CLASS_IDS_200

parser = ArgumentParser()
parser.add_argument('--inp', type=str)
parser.add_argument('--out', type=str)
args = parser.parse_args()

with open(args.inp) as f:
    expr_dct:dict = json.load(f)
model = SentenceTransformer('all-MiniLM-L12-v2').cuda()

result = []
for label in tqdm(CLASS_LABELS_200):
    embeddings = model.encode(expr_dct[label], convert_to_tensor=True, )
    # result[label] = embeddings
    result.append(embeddings)

result = torch.stack(result)
torch.save(result, args.out)
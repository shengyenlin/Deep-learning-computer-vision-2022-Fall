# Explaination-based Pretraining

## Environment

* ubuntu 20.04
* conda environment
```bash
conda env create -f environment_pretrain.yml
conda activate ocnn
```

### library modification

* add two methods to `ocnn.models.UNet`
```py
@property
def feature_dim(self):
    return self.decoder_channel[-1]

def forward_features(self, data: torch.Tensor, octree: Octree, depth: int,
              query_pts: torch.Tensor):
    r''''''

    convd = self.unet_encoder(data, octree, depth)
    deconv = self.unet_decoder(convd, octree, depth - self.encoder_stages)

    interp_depth = depth - self.encoder_stages + self.decoder_stages
    feature = self.octree_interp(deconv, octree, interp_depth, query_pts)
    return feature

```

## Obtain Explaination Embeddings

* set `openai.api_key='<api_key>'` in label_text/generate_explaination.ipynb and run this ipynb.
* run `python label_text/generate_embeddings.py --inp <explaiantion file> --out <target_embeddings>`

## run-pretraining

* edit config in `configs/simsiam_pretrain_dual.yaml`
     * data path
     * gpu id
     * batch size
* run

```sh
python simsiam_pretrain_lightning.py --config configs/simsiam_pretrain_dual.yaml
```

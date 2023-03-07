import torch
import ocnn
from thsolver import get_config
from thsolver.config import parse_args
from thsolver.lr_scheduler import get_lr_scheduler
import pytorch_lightning as pl
from datasets import get_scannet_dataset
from torch import nn
import torch.nn.functional as F
from pretrain_module import MLP, simsiam_loss, RandomLabelEmbeddings, AttentionLabelEmbedding, WeightBy3DLabelEmbedding

def accuracy(logit, label):
    pred = logit.argmax(dim=1)
    accu = pred.eq(label).float().mean()
    return accu

def IoU_per_shape(logit: torch.Tensor, label: torch.Tensor, class_num):
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
    return IoU.item()

class SimsiamWrapper(nn.Module):
    def __init__(self, embed_size, out_size):
        """
        
        """
        super().__init__()
        self.projector = MLP(embed_size, embed_size, out_size, 2)
        self.projector.add_module('batch_norm',
                                  nn.BatchNorm1d(out_size, affine=False))
        self.predictor = MLP(embed_size, embed_size, out_size, 2)

    def forward(self, feature_3d: torch.Tensor,
                feature_text: torch.Tensor) -> torch.Tensor:
        """
        return loss
        """
        proj_3d = self.projector(feature_3d)
        proj_text = self.projector(feature_text)

        pred_3d = self.predictor(proj_3d)
        pred_text = self.predictor(proj_text)
        return simsiam_loss(pred_3d, pred_text, proj_3d.detach(),
                            proj_text.detach())

    def forward_feature_3d(self, feature_3d_1: torch.Tensor, feature_3d_2: torch.Tensor):
        z1, z2 = self.projector(feature_3d_1), self.projector(feature_3d_2)
        p1, p2 = self.predictor(z1), self.projector(z2)
        return simsiam_loss(p1, p2, z1.detach(), z2.detach())
    
class SimsiamPLModule(pl.LightningModule):
    def __init__(self, flags) -> None:
        super().__init__()
        if isinstance(flags, dict):
            from argparse import Namespace
            flags = Namespace(**flags)
        self.flags = flags
        self.save_hyperparameters(flags)
        if self.flags.MODEL.name.lower() == 'unet':
            model_fl = flags.MODEL
            self.model = ocnn.models.UNet(model_fl.channel, model_fl.nout, model_fl.interp,
                                     model_fl.nempty)
        else:
            raise NotImplementedError
        
        self.label_embeddings = self.get_label_embedding()
        self.wrapper = SimsiamWrapper(self.model.feature_dim,
                                      self.model.feature_dim)
    
    def get_label_embedding(self) -> nn.Module:
        embedding: nn = {
            'rand': RandomLabelEmbeddings,
            'attn': AttentionLabelEmbedding,
            'weight3d': WeightBy3DLabelEmbedding
        }[self.flags.MODEL.label_embedding_type]
        return embedding(
            self.model.feature_dim, self.flags.MODEL.label_embedding_path,
        True)
    
    def configure_optimizers(self):
        flags = self.flags.SOLVER
        base_lr = flags.lr
        parameters = [{
            'params': self.model.parameters()
        }, {
            'params': self.label_embeddings.parameters()
        }, {
            'params': self.wrapper.parameters()
        }]

        # config the optimizer
        if flags.type.lower() == 'sgd':
            optimizer = torch.optim.SGD(parameters,
                                             lr=base_lr,
                                             weight_decay=flags.weight_decay,
                                             momentum=0.9)
        elif flags.type.lower() == 'adam':
            optimizer = torch.optim.Adam(parameters,
                                              lr=base_lr,
                                              weight_decay=flags.weight_decay)
        elif flags.type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(parameters,
                                               lr=base_lr,
                                               weight_decay=flags.weight_decay)
        else:
            raise ValueError
        scheduler = get_lr_scheduler(optimizer, self.flags.SOLVER)
        return [optimizer], [scheduler]
    
    def forward(self, batch):
        """
        points.labels: [points]
        """
        octree = batch['octree']
        points = batch['points']
        flags = self.flags.MODEL
        octree_feature = ocnn.modules.InputFeature(flags.feature, flags.nempty)
        data = octree_feature(octree)
        query_pts = torch.cat([points.points, points.batch_id], dim=1)

        features_3d = self.model.forward_features(data, octree, octree.depth,
                                                  query_pts)  # points, E
        label_mask = points.labels > self.flags.LOSS.mask  # filter labels

        features_3d = features_3d[label_mask]  # points, 96
        labels = points.labels[label_mask]  # points

        return features_3d, labels.long()
    
    def training_step(self, batch, idx):
        batch1, batch2 = batch
        features_3d_1, labels = self(batch1)
        features_3d_2, labels = self(batch2)
        feature_text = self.label_embeddings(labels, features_3d_1.detach())
        loss_3d_text = self.wrapper(features_3d_2, feature_text)
        loss_3d_3d = self.wrapper.forward_feature_3d(features_3d_1, features_3d_2)
        loss = .5 * (loss_3d_3d + loss_3d_text)
        self.log_dict({ 'train/loss': loss, 
                       'train/loss_3d_3d': loss_3d_3d, 
                       'train/loss_3d_text': loss_3d_text
                        }, sync_dist=True, batch_size=idx)
        if idx % 10 == 0:
            torch.cuda.empty_cache()
        return loss
    
    def training_epoch_end(self, outputs) -> None:
        torch.cuda.empty_cache()
    
    def export_model(self, path: str):
        torch.save(
            self.model.state_dict(), path
        )
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        features_3d, labels = self(batch) # [N, 96], [N]
        all_labels = torch.arange(0, 201, device=features_3d.device)
        all_label_embeds = self.label_embeddings(all_labels, None) # [201, 96]
        feature_text = self.label_embeddings(labels, features_3d)
        sim = torch.cosine_similarity(features_3d, feature_text, dim=-1)
        features_3d, all_label_embeds = F.normalize(features_3d, dim=-1), F.normalize(all_label_embeds, dim=-1)
        score = features_3d.matmul(all_label_embeds.t()) # N, 201
        score[:, 0] = -100
        pred_labels = score.argmax(-1) # [N]
        count = (labels == pred_labels).float()
        miou = IoU_per_shape(score, labels, 201)
        return {'val/cos_sim': sim.cpu(), 'val/acc': count.cpu(), 'val/mIOU': miou}
    
    def validation_epoch_end(self, outputs) -> None:
        sim = torch.cat([o['val/cos_sim'] for o in outputs]).reshape(-1)
        acc = torch.cat([o['val/acc'] for o in outputs]).reshape(-1)
        mIOU = torch.tensor([o['val/mIOU'] for o in outputs]).reshape(-1)
        self.log('val/mean_cos_sim', sim.mean(), sync_dist=True, prog_bar=True)
        self.log('val/acc', acc.mean(), sync_dist=True, prog_bar=True)
        self.log('val/mIOU', mIOU.mean(), sync_dist=True, prog_bar=True)

    @classmethod
    def update_configs(cls):
        FLAGS = get_config()
        FLAGS.LOSS.mask = 0  # mask the invalid labels
        FLAGS.LOSS.accumulation = 4
        FLAGS.MODEL.label_embedding_path = ''
        FLAGS.MODEL.label_embedding_type = 'attn'  # attn or rand
        FLAGS.SOLVER.amp = 'native'  # attn or rand
        FLAGS.SOLVER.precision = 16 
        FLAGS.resume = ''  


# CUDA_VISIBLE_DEVICES=0 nohup python simsiam_pretrain.py --config configs/simsiam_pretrain_scannet.yaml > simsiam.out&
if __name__ == "__main__":
    from pretrain_module import get_dual_pretraining_dataloader
    SimsiamPLModule.update_configs()
    FLAGS = parse_args()
    pl.seed_everything(42)
    
    val_ds, val_collate_fn = get_scannet_dataset(FLAGS.DATA.test)
    print(FLAGS)
    train_dl = get_dual_pretraining_dataloader(FLAGS.DATA.train)
    val_dl = torch.utils.data.DataLoader(
    val_ds, batch_size=FLAGS.DATA.test.batch_size, num_workers=FLAGS.DATA.test.num_workers,
     collate_fn=val_collate_fn, pin_memory=False)
    model = SimsiamPLModule(FLAGS)
    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=FLAGS.SOLVER.max_epoch,
        strategy="ddp",
        accumulate_grad_batches=FLAGS.LOSS.accumulation,
        amp_backend=FLAGS.SOLVER.amp,
        sync_batchnorm=True,
        precision=FLAGS.SOLVER.precision,
        devices=FLAGS.SOLVER.gpu,
        enable_checkpointing=True,
        check_val_every_n_epoch=FLAGS.SOLVER.test_every_epoch,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='val/acc',
                save_top_k=5,
                mode='max',
                save_last=True
            )
        ]
    )
    trainer.fit(model, train_dl, val_dl, ckpt_path=FLAGS.resume)
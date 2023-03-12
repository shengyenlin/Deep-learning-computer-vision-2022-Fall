import gc
import numpy as np
from tqdm import tqdm
from statistics import mean

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F
from torch.autograd import Variable

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(
            x, 
            size = self.size,
            mode = self.mode,
            align_corners=False
        )

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = 0
    for i in range(6): #maybe there's no one class in this pic
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        if (tp_fp + tp_fn - tp) == 0:
            iou = 0
        else:
            iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
    #     print('class #%d : %1.5f'%(i, iou))
    # print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou

def do_avg_on_list(list):
    return round(mean(list), 2)

def save_checkpoint(model, optimizer, valid_acc, prefix):
    checkpoint_path = f'{prefix}.pth'
    torch.save({
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'val_iou': valid_acc
    },checkpoint_path)

def train_P2_segform(train_loader, model, loss_fn, optimizer, device):
    train_loss = 0
    n_batch = 0
    model.train()
    gc.collect()
    torch.cuda.empty_cache()
    for batch in tqdm(train_loader, leave=False, colour='green'):
        img, label = batch['pixel_values'], batch['labels']
        img, label = img.to(device), label.to(device)
        n_mini_batch = img.size(0)
        optimizer.zero_grad()
        _, logits = model(img, label)
        
        upsampled_logits = nn.functional.interpolate(
                logits,
                size=label.shape[-2:], # (height, width)
                mode='bilinear',
                align_corners=False
            )
        loss = loss_fn(upsampled_logits, label)

        loss.backward()            
        optimizer.step()
        
        with torch.no_grad():

            #print(label.shape[-2:])
            predict = upsampled_logits.argmax(dim=1)
            predict = predict.detach().cpu().numpy() #numpy array
            loss = loss.item() #python float
            train_loss += loss * n_mini_batch
            n_batch += n_mini_batch
        del img, loss, predict
        torch.cuda.empty_cache()

    train_loss = train_loss / n_batch
    return train_loss

def valid_P2_segform(valid_loader, model, loss_fn, device):
    is_first_mini_batch = True
    model.eval()
    valid_loss = 0
    n_batch = 0
    with torch.no_grad():
        for batch in valid_loader:
            img, label = batch['pixel_values'], batch['labels']
            img, label = img.to(device), label.to(device)
            n_mini_batch = img.size(0)

            _, logits = model(img, label)
            
            upsampled_logits = nn.functional.interpolate(
                logits,
                size=label.shape[-2:], # (height, width)
                mode='bilinear',
                align_corners=False
            )
            
            loss = loss_fn(upsampled_logits, label)
            predict = upsampled_logits.argmax(dim=1)

            predict = predict.detach().cpu().numpy()
            #print(predict.shape) #(n_b, 512, 512)
            label = label.detach().cpu().numpy()
            #collect iou and label
            if is_first_mini_batch:
                iou_matrix = predict
                label_matrix = label
                is_first_mini_batch = False
            else:
                iou_matrix = np.concatenate(
                    [iou_matrix, predict], axis = 0
                )
                label_matrix = np.concatenate(
                    [label_matrix, label], axis = 0
                )
            loss = loss.item()
            valid_loss += loss * n_mini_batch
            n_batch += n_mini_batch
            del img, label, loss, predict
            torch.cuda.empty_cache()
            
    # print(iou_matrix.shape)
    # print(label_matrix.shape)
    valid_iou = mean_iou_score(iou_matrix, label_matrix)
    valid_loss = valid_loss / n_batch
    gc.collect()
    torch.cuda.empty_cache()
    return valid_iou, valid_loss

def train_P2(train_loader, model, loss_fn, optimizer, device):
    # is_first_mini_batch = True
    train_loss = 0
    n_batch = 0
    model.train()
    gc.collect()
    torch.cuda.empty_cache()
    for img, label in tqdm(train_loader, leave=False, colour='green'):
        n_mini_batch = img.size(0)
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = loss_fn(output, label)
        loss.backward()            
        optimizer.step()
        
        with torch.no_grad():
            predict = torch.argmax(output, dim=1)
            predict = predict.detach().cpu().numpy() #numpy array
            label = label.detach().cpu().numpy() 
            loss = loss.item() #python float
            train_loss += loss * n_mini_batch
            n_batch += n_mini_batch
        del img, label, output, loss, predict
        torch.cuda.empty_cache()
    # train_iou = mean_iou_score(iou_matrix, label_matrix)
    train_loss = train_loss / n_batch
    return train_loss


def valid_P2(valid_loader, model, loss_fn, device):
    is_first_mini_batch = True
    model.eval()
    valid_loss = 0
    n_batch = 0
    with torch.no_grad():
        for img, label in valid_loader:
            img, label = img.to(device), label.to(device)
            n_mini_batch = img.size(0)

            output = model(img) 
            loss = loss_fn(output, label)
            predict = torch.argmax(output, dim=1)
            predict = predict.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            
            #collect iou and label
            if is_first_mini_batch:
                iou_matrix = predict
                label_matrix = label
                is_first_mini_batch = False
            else:
                iou_matrix = np.concatenate(
                    [iou_matrix, predict], axis = 0
                )
                label_matrix = np.concatenate(
                    [label_matrix, label], axis = 0
                )
            

            loss = loss.item()
            valid_loss += loss * n_mini_batch
            n_batch += n_mini_batch
            del img, label, output, loss, predict
            torch.cuda.empty_cache()
            
    valid_iou = mean_iou_score(iou_matrix, label_matrix)
    valid_loss = valid_loss / n_batch
    gc.collect()
    torch.cuda.empty_cache()
    return valid_iou, valid_loss

def train(train_loader, model, loss_fn, optimizer, device):
    train_loss = []
    train_acc = []
    model.train()
    for img, label in tqdm(train_loader, leave=False, colour='green'):
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = loss_fn(output, label)
        loss.backward()            
        optimizer.step()
        with torch.no_grad():
            predict = torch.argmax(output, dim=-1)
            acc = np.mean(
                (label == predict).cpu().numpy()
            )
            train_acc.append(acc)
            train_loss.append(loss.item())
        del img, label, output, loss, predict, acc

    train_acc = np.mean(train_acc)
    train_loss = np.mean(train_loss)
    return train_acc, train_loss
    
def valid(valid_loader, model, loss_fn, device):
    model.eval()
    with torch.no_grad():
        valid_loss = []
        valid_acc = []
        for img, label in valid_loader:
            img, label = img.to(device), label.to(device)
            output = model(img)
            loss = loss_fn(output, label)
            predict = torch.argmax(output, dim=-1)
            acc = (label == predict).cpu().tolist()
            valid_loss.append(loss.item())
            valid_acc += acc
            del img, label, output, loss, predict, acc

    valid_acc = np.mean(valid_acc)
    valid_loss = np.mean(valid_loss)

    return valid_acc, valid_loss


class P1Args:
    def __init__(self):
        #basic
        self.batch_size = 48
        self.patience = 5
        self.num_epoch = 600
        self.num_workers = 8
        self.k_folds = 3
        self.model_name = '1006_modelB'

        #params of optimizer
        self.optim_name = 'SGD'
        self.momentum=0
        self.lr=0.0001     
        self.weight_decay = 0      

        #params of scheduler
        self.lr_scheduler='cosineannealinglr' 
        self.lr_warmup_epochs=5 
        self.lr_warmup_method='linear' 
        self.lr_warmup_decay=0.01

        #loss function
        self.label_smoothing = 0.1

        #optimizer and scheduler
        self.optimizer = None
        self.scheduler = None

    def setup_optimizer(
        self, model, optim_name, 
        lr, momentum, weight_decay=2e-05
        ):
        if optim_name == 'SGD':
            optimizer = getattr(optim, self.optim_name)(
                model.parameters(), 
                lr=lr, 
                momentum=momentum, 
                weight_decay=weight_decay
            )
        elif optim_name == 'Adam':
            optimizer = getattr(optim, self.optim_name)(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        self.optimizer = optimizer
        return optimizer


    def setup_scheduler(self):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=self.num_epoch - self.lr_warmup_epochs,
            # lr_warmup_method=lr_warmup_method,
            # lr_warmup_decay=lr_warmup_decay
        )
        self.scheduler = scheduler
        return scheduler

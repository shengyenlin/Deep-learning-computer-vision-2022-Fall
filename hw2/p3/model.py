import torch.nn as nn
from torch.autograd import Function


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DANN(nn.Module):
    def __init__(self, use_dann, get_feat=None):
        super(DANN, self).__init__()
        self.use_dann = use_dann
        self.get_feat = get_feat

        # bo's model
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3,64,5), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2),  
            nn.Conv2d(64,64,5),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,128,5),
            )  

        self.domain_clf = nn.Sequential(
                    nn.Linear(1152,1024),
                     nn.ReLU(inplace=True),
                     nn.Linear(1024,1024),
                     nn.ReLU(inplace=True),
                     nn.Linear(1024,2),
                    #  nn.Sigmoid()
                     )

        self.label_clf = nn.Sequential(
            nn.Linear(1152,3072),
            nn.ReLU(inplace=True),
            nn.Linear(3072,2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048,10),
            nn.LogSoftmax(dim=1)
        )

        # self.feature_extractor = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=32,
        #               kernel_size=(5, 5)),  # 3 28 28, 32 24 24
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=(2, 2)),  # 32 12 12
        #     nn.Conv2d(in_channels=32, out_channels=48,
        #               kernel_size=(5, 5)),  # 48 8 8
        #     nn.BatchNorm2d(48),
        #     nn.Dropout2d(),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=(2, 2)),  # 48 4 4
        # )

        # self.domain_clf = nn.Sequential(
        #     nn.Linear(48*4*4, 100),
        #     nn.BatchNorm1d(100),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(100, 1),
        #     nn.Sigmoid()
        # )    

        # self.label_clf = nn.Sequential(
        #     nn.Linear(48*4*4, 100),
        #     nn.BatchNorm1d(100),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(100, 100),
        #     nn.BatchNorm1d(100),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(100, 10),
        #     nn.LogSoftmax(dim=1)
        # )

    def forward(self, x, lam=None):
        # (bs, c, h, w) = (bs, 3, 28, 28)
        # TODO: check why we need .expand
        x = x.expand(x.data.shape[0], 3, 28, 28)
        feat = self.feature_extractor(x)
        feat = feat.view(x.shape[0], -1) # flatten feature
        class_out = self.label_clf(feat)

        if self.use_dann:
            reverse_feat = ReverseLayerF.apply(feat, lam)
            domain_out = self.domain_clf(reverse_feat)
            if self.get_feat:
                return feat, class_out
            else:
                return class_out, domain_out

        else:
            return class_out
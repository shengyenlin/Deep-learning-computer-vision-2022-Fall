from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models.segmentation.deeplabv3 import DeepLabV3, DeepLabHead
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class P1_ModelA(nn.Module):
    def __init__(self, num_class):
        super(P1_ModelA, self).__init__() #3x64x64
        self.num_class = num_class
        ###########################general information###########################
        self.conv1_out_channels = 64
        self.leakyReLU1_slope = 0.05
        self.conv1_dropout_rate = 0.2

        self.conv0 = nn.Sequential(
            nn.Conv2d(3, self.conv1_out_channels, kernel_size=5, padding=2), #64x64x64
            nn.LeakyReLU(negative_slope=self.leakyReLU1_slope),
            nn.BatchNorm2d(self.conv1_out_channels),
            nn.Dropout2d(0.1)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.conv1_out_channels, self.conv1_out_channels, kernel_size=5, padding=2), #64x64x64
            nn.LeakyReLU(negative_slope=self.leakyReLU1_slope),
            nn.BatchNorm2d(self.conv1_out_channels),
            nn.MaxPool2d(2), #64x32x32
            nn.Dropout2d(0.1)
        )

        #########################location information######################
        self.conv2_out_channels = 512
        self.leakyReLU2_slope = 0.05
        self.conv2_dropout_rate = 0.5
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.conv1_out_channels, 128, kernel_size=3, padding=1), #128x32x32
            nn.LeakyReLU(negative_slope=self.leakyReLU2_slope),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2), #128x16x16
            nn.Dropout2d(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, self.conv2_out_channels, kernel_size=3, padding=1), #128x16x16
            nn.LeakyReLU(negative_slope=self.leakyReLU2_slope),
            nn.BatchNorm2d(self.conv2_out_channels),
            nn.MaxPool2d(2), #512x8x8
            nn.Dropout2d(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.conv2_out_channels, self.conv2_out_channels, kernel_size=3,padding=1), #128x8x8
            nn.LeakyReLU(negative_slope=self.leakyReLU2_slope),
            nn.BatchNorm2d(self.conv2_out_channels),
            nn.MaxPool2d(2), #512x4x4
            nn.Dropout2d(0.2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(4*4*self.conv2_out_channels, self.conv2_out_channels),
            nn.ReLU(),
            nn.BatchNorm1d(self.conv2_out_channels),
            nn.Dropout(0.3)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.conv2_out_channels, self.conv2_out_channels),
            nn.ReLU(),
            nn.BatchNorm1d(self.conv2_out_channels),
            nn.Dropout(0.3)
        )
        self.fc3 = nn.Linear(self.conv2_out_channels, self.num_class)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 4*4*self.conv2_out_channels)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

######P2#####
class VGG16_FCN32(torch.nn.Module):
    def __init__(self, n_classes):
        super(VGG16_FCN32, self).__init__()
        self.features = models.vgg16(weights='IMAGENET1K_V1').features 
        self.fc6 = nn.Sequential(
            nn.Conv2d(512, 4096, 2), #kernel = 7x7?
            nn.ReLU(inplace=True),
            nn.Dropout() #dropout2d?
        )
        self.fc7 = nn.Sequential(
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout() #dropout2d?
        )
        self.clf = nn.Conv2d(4096, n_classes, 1)
        self.upconv = nn.ConvTranspose2d(
            n_classes, n_classes, kernel_size=64, stride=32, bias=False
            )

    def forward(self, x): #3x512x512
        x = self.features(x) #512x16x16
        x = self.fc6(x) #4096x15x15
        x = self.fc7(x) #4096x15x15
        x = self.clf (x) #7x15x15
        x = self.upconv(x) #7x512x512 (upsample)
        return x

class VGG16_FCN16(nn.Module):
    def __init__(self, n_classes):
        super(VGG16_FCN16, self).__init__() #3*256*256
        self.vgg = models.vgg16(weights='IMAGENET1K_V1') 
        del self.vgg.avgpool
        del self.vgg.classifier
        self.pool4 = self.vgg.features[:24] #512*16*16
        self.pool5 = self.vgg.features[24:] #512*8*8
        self.fc = nn.Sequential(
            nn.Conv2d(512,4096,2), #4096*7*7
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096,4096,1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096,n_classes,1), #7*7*7
            nn.ConvTranspose2d(n_classes,512,4,2,bias=False) #512*16*16
            )
        self.pred = nn.ConvTranspose2d(512,n_classes,16,16,bias=False)
    def forward (self, x) :        
        pool4_output = self.pool4(x) #512*16*16
        pool5_output = self.pool5(pool4_output) #512*8*8
        x = self.fc(pool5_output) #512*16*16
        x = self.pred(x+pool4_output)
        return x

class VGG16_FCN8(nn.Module):
    def __init__(self, n_classes):
        super(VGG16_FCN8, self).__init__() #3*256*256
        self.vgg = models.vgg16(weights='IMAGENET1K_V1') 
        del self.vgg.avgpool
        del self.vgg.classifier
        self.pool3 = self.vgg.features[:17] #256*32*32
        self.pool4 = self.vgg.features[17:24] #512*16*16
        self.pool5 = self.vgg.features[24:] #512*8*8
        self.fc_pool5 = nn.Sequential(nn.Conv2d(512,4096,2), #4096*7*7
                               nn.ReLU(inplace=True),
                               nn.Dropout2d(),
                               nn.Conv2d(4096,4096,1),
                               nn.ReLU(inplace=True),
                               nn.Dropout2d(),
                               nn.Conv2d(4096,n_classes,1), #7*7*7
                               nn.ConvTranspose2d(n_classes,256,8,4,bias=False) #256*32*32
                               )
        self.fc_pool4 = nn.ConvTranspose2d(512,256,2,2,bias=False) #256*32*32
        self.pred = nn.ConvTranspose2d(256,n_classes,8,8,bias=False)
    def forward (self, x):        
        pool3_output = self.pool3(x) #256*32*32
        pool4_output = self.pool4(pool3_output) #512*16*16
        pool5_output = self.pool5(pool4_output) #512*8*8
        x = self.pred(self.fc_pool5(pool5_output) + self.fc_pool4(pool4_output) + pool3_output)        
        return x



# handcraft segformer
# class SegFormer(nn.Module):
#     def __init__(self, num_class):
#         super(SegFormer, self).__init__()
#         self.feature_extractor, self.model = get_segformer(
#             num_class
#         )
#     self.upsample = Interpolate(
#         size = (512, 512),
#         mode = 'bilinear'
#     )

#     def forward(self, x):
#         x = self.feature_extractor(x, return_tensors = 'pt')
#         x = self.model(**x)
#         logits = x.logits
#         upsampled_logits = nn.functional.interpolate(
#             logits,
#             size=(512, 512), # (height, width)
#             mode='bilinear',
#             align_corners=False
#             )
#         return upsampled_logits

class SegFormer(nn.Module):
    def __init__(self, num_class, seg_type = 'b1', mode='train'):
        super(SegFormer, self).__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            f"nvidia/segformer-{seg_type}-finetuned-ade-512-512", 
            num_labels = num_class,
            ignore_mismatched_sizes=True, 
            return_dict=False,
            )
        self.mode = mode

    def forward(self, images, masks=None):
        if self.mode == 'test':
            x = self.model(pixel_values=images)
        else:
            x = self.model(pixel_values=images, labels=masks)
        return x

def get_deep_lab_resnet101(num_class):
    backbone = models.resnet.resnet101(
        weights='ResNet101_Weights.IMAGENET1K_V2', 
        replace_stride_with_dilation=[False, True, True]
    )
    aux = True
    aux_classifier = FCNHead(1024, num_class) if aux else None
    classifier = DeepLabHead(2048, num_class)
    model = DeepLabV3(backbone, classifier, aux_classifier)
    return model

# def get_segformer(num_class):
#     feature_extractor = SegformerFeatureExtractor.from_pretrained(
#         "nvidia/segformer-b1-finetuned-ade-512-512"
#         )
#     model = SegformerForSemanticSegmentation.from_pretrained(
#         "nvidia/segformer-b1-finetuned-ade-512-512"
#         )
#     model.decode_head.classifier = nn.Conv2d(
#         256, num_class, 
#         kernel_size=(1,1), stride=(1,1)
#     )
#     return feature_extractor, model

def get_p2_model(model_name, num_class, seg_type=None, mode='train'):
    if model_name == 'deeplabl_resnet101':
        model = get_deep_lab_resnet101(num_class)
        return model

    elif model_name == 'VGG16_FCN32':
        model = VGG16_FCN32(num_class)
        return model
    
    elif model_name == 'VGG16_FCN16':
        model = VGG16_FCN8(num_class)
        return model

    elif model_name == 'VGG16_FCN8':
        model = VGG16_FCN8(num_class)
        return model

    elif model_name == 'segformer':
        model = SegFormer(num_class, seg_type, mode)
        return model
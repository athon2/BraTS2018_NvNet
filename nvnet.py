import os 
# os.environ["CUDA_VISIABLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F

class DownSampling(nn.Module):
    # 3x3x3 convolution,1 padding as default
    def __init__(self, inChans, outChans, stride=2, kernel_size=3, padding=1):
        super(DownSampling, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels=inChans, 
                     out_channels=outChans, 
                     kernel_size=kernel_size, 
                     stride=stride,
                     padding=padding,
                     bias=False)
    def forward(self, x):
        return self.conv1(x)
    
class EncoderBlock(nn.Module):
    def __init__(self, inChans, outChans, stride=1, padding=1, num_groups=4, activation="relu", normalizaiton="group_normalization"):
        super(EncoderBlock, self).__init__()
        
        if normalizaiton == "group_normalization":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=inChans)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=inChans)
        if activation == "relu":
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=3, stride=stride, padding=padding)
        self.conv2 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=3, stride=stride, padding=padding)
        
        
    def forward(self, x):
        residual = x
        
        out = self.norm1(x)
        out = self.actv1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.actv2(out)
        out = self.conv2(out)
        
        out += residual
        return out
    
class LinearUpSampling(nn.Module):
    # Trilinear 
    def __init__(self, inChans, outChans, scale_factor=2, mode="trilinear", align_corners=True):
        super(LinearUpSampling, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=1)
 
        self.up1 = nn.Upsample(scale_factor=scale_factor,mode=mode, align_corners=align_corners)
        self.conv2 = nn.Conv3d(in_channels=outChans, out_channels=outChans, kernel_size=1)
    
    def forward(self, x, skipx=None):
        out = self.conv1(x)
        out = self.up1(out)
        if skipx is not None:
            out = torch.cat((out, skipx), 1)
            out = self.conv2(out)
        
        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, inChans, outChans, stride=1, padding=1, num_groups=4, activation="relu", normalizaiton="group_normalization"):
        super(DecoderBlock, self).__init__()
        
        if normalizaiton == "group_normalization":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=outChans)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=outChans)
        if activation == "relu":
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=3, stride=stride, padding=padding)
        self.conv2 = nn.Conv3d(in_channels=outChans, out_channels=outChans, kernel_size=3, stride=stride, padding=padding)
        
        
    def forward(self, x):
        residual = x
        
        out = self.norm1(x)
        out = self.actv1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.actv2(out)
        out = self.conv2(out)
        
        out += residual
        return out
    
class OutputTransition(nn.Module):
    def __init__(self, inChans, outChans):
        super(OutputTransition, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=1)
        self.actv1 = F.sigmoid
        
    def forward(self, x):
        return self.actv1(self.conv1(x))

class VDResampling(nn.Module):
    def __init__(self, inChans=256, outChans=256, stride=2, kernel_size=3, padding=1, activation="relu", normalizaiton="group_normalization"):
        super(VDResampling, self).__init__()
        midChans = int(inChans / 2)
        if normalizaiton == "group_normalization":
            self.gn1 = nn.GroupNorm(num_groups=4,num_channels=inChans)
        if activation == "relu":
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=16, kernel_size=kernel_size, stride=stride, padding=padding)
        self.dense1 = nn.Linear(in_features=16*10*12*8, out_features=inChans)
        
        self.dense2 = nn.Linear(in_features=midChans, out_features=midChans*10*12*8)

        self.up0 = LinearUpSampling(midChans,outChans)
        
    def forward(self, x):
        out = self.gn1(x)
        out = self.actv1(out)
        out = self.conv1(out)
        out = out.view(-1, self.num_flat_features(out))
        out = self.dense1(out)
        out = VDraw(out)
        out = self.dense2(out)
        out = self.actv2(out)
        out = out.view((1, 128, 10, 12, 8))
        out = self.up0(out)
        
        return out
        
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def VDraw(x):
    return torch.distributions.Normal(x[:,:128], x[:,128:]).sample()

class VDecoderBlock(nn.Module):
    def __init__(self, inChans, outChans, activation="relu", normalizaiton="group_normalization", mode="trilinear"):
        super(VDecoderBlock, self).__init__()

        self.up0 = LinearUpSampling(inChans, outChans, mode=mode)
        self.block = DecoderBlock(outChans, outChans, activation=activation, normalizaiton=normalizaiton)
    
    def forward(self, x):
        out = self.up0(x)
        out = self.block(out)

        return out

class VAE(nn.Module):
    def __init__(self, inChans=256, outChans=4, activation="relu", normalizaiton="group_normalization", mode="trilinear"):
        super(VAE, self).__init__()
        self.vd_resample = VDResampling(inChans, inChans)
        self.vd_block2 = VDecoderBlock(inChans, inChans//2)
        self.vd_block1 = VDecoderBlock(inChans//2, inChans//4)
        self.vd_block0 = VDecoderBlock(inChans//4, inChans//8)
        self.vd_end = nn.Conv3d(inChans//8, outChans, kernel_size=1)
    def forward(self, x):
        out = self.vd_resample(x)
        out = self.vd_block2(out)
        out = self.vd_block1(out)
        out = self.vd_block0(out)
        out = self.vd_end(out)

        return out

class NvNet(nn.Module):
    def __init__(self, inChans=4, seg_outChans=1, activation="relu", normalizaiton="group_normalization", mode="trilinear"):
        super(NvNet, self).__init__()
        self.in_conv0 = DownSampling(inChans=inChans, outChans=32, stride=1)
        self.en_block0 = EncoderBlock(32, 32, activation=activation, normalizaiton=normalizaiton)
        self.en_down1 = DownSampling(32, 64)
        self.en_block1_0 = EncoderBlock(64, 64, activation=activation, normalizaiton=normalizaiton)
        self.en_block1_1 = EncoderBlock(64, 64, activation=activation, normalizaiton=normalizaiton)
        self.en_down2 = DownSampling(64, 128)
        self.en_block2_0 = EncoderBlock(128, 128, activation=activation, normalizaiton=normalizaiton)
        self.en_block2_1 = EncoderBlock(128, 128, activation=activation, normalizaiton=normalizaiton)
        self.en_down3 = DownSampling(128, 256)
        self.en_block3_0 = EncoderBlock(256, 256, activation=activation, normalizaiton=normalizaiton)
        self.en_block3_1 = EncoderBlock(256, 256, activation=activation, normalizaiton=normalizaiton)
        self.en_block3_2 = EncoderBlock(256, 256, activation=activation, normalizaiton=normalizaiton)
        self.en_block3_3 = EncoderBlock(256, 256, activation=activation, normalizaiton=normalizaiton)
        
        self.de_up2 =  LinearUpSampling(256, 128, mode=mode)
        self.de_block2 = DecoderBlock(128, 128, activation=activation, normalizaiton=normalizaiton)
        self.de_up1 =  LinearUpSampling(128, 64, mode=mode)
        self.de_block1 = DecoderBlock(64, 64, activation=activation, normalizaiton=normalizaiton)
        self.de_up0 =  LinearUpSampling(64, 32, mode=mode)
        self.de_block0 = DecoderBlock(32, 32, activation=activation, normalizaiton=normalizaiton)
        self.de_end = OutputTransition(32, seg_outChans)
        
        self.vae = VAE(256, inChans)

    def forward(self, x):
        out_init = self.in_conv0(x)
        out_en0 = self.en_block0(out_init)
        out_en1 = self.en_block1_1(self.en_block1_0(self.en_down1(out_en0))) 
        out_en2 = self.en_block2_1(self.en_block2_0(self.en_down2(out_en1)))
        out_en3 = self.en_block3_3(
            self.en_block3_2(
                self.en_block3_1(
                    self.en_block3_0(
                        self.en_down3(out_en2)))))
        out_de2 = self.de_block2(self.de_up2(out_en3, out_en2))
        out_de1 = self.de_block1(self.de_up1(out_de2, out_en1))
        out_de0 = self.de_block0(self.de_up0(out_de1, out_en0))

        out_end = self.de_end(out_de0)
        out_vae = self.vae(out_en3)
        out_final = torch.cat((out_end, out_vae), 1)

        return out_final
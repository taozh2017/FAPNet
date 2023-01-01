import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from lib.Res2Net_v1b import res2net50_v1b_26w_4s


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    

###################################################################    
class MFAM0(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(MFAM0, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)

  
        self.conv_1_1 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_2 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_3 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_4 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_5 = BasicConv2d(out_channels, out_channels, 3, stride=1, padding=1)
        
        self.conv_3_1 = nn.Conv2d(out_channels,   out_channels , kernel_size=3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.conv_5_1 = nn.Conv2d(out_channels,   out_channels , kernel_size=5, stride=1, padding=2)
        self.conv_5_2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        

    def forward(self, x):
        
        ###+
        x1     = x # self.conv_1_1(x)
        x2     = x # self.conv_1_2(x)
        x3     = x # self.conv_1_3(x)
        
        x_3_1  = self.relu(self.conv_3_1(x2))  ## (BS, 32, ***, ***)
        x_5_1  = self.relu(self.conv_5_1(x3))  ## (BS, 32, ***, ***)
        
        x_3_2 = self.relu(self.conv_3_2(x_3_1 + x_5_1))  ## (BS, 64, ***, ***)
        x_5_2 = self.relu(self.conv_5_2(x_5_1 + x_3_1))  ## (BS, 64, ***, ***)
         
        x_mul = torch. mul(x_3_2, x_5_2)
        out   = self.relu(x1 + self.conv_1_5(x_mul + x_3_1 + x_5_1))

        return out
    
class MFAM(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(MFAM, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)

        self.conv_1_1 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_2 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_3 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_4 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_5 = BasicConv2d(out_channels, out_channels, 3, stride=1, padding=1)
        
        self.conv_3_1 = nn.Conv2d(out_channels,   out_channels , kernel_size=3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.conv_5_1 = nn.Conv2d(out_channels,   out_channels , kernel_size=5, stride=1, padding=2)
        self.conv_5_2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        

    def forward(self, x):
        
        ###+
        x1     = self.conv_1_1(x)
        x2     = self.conv_1_2(x)
        x3     = self.conv_1_3(x)
        
        x_3_1  = self.relu(self.conv_3_1(x2))  ## (BS, 32, ***, ***)
        x_5_1  = self.relu(self.conv_5_1(x3))  ## (BS, 32, ***, ***)
        
        x_3_2  = self.relu(self.conv_3_2(x_3_1 + x_5_1))  ## (BS, 64, ***, ***)
        x_5_2  = self.relu(self.conv_5_2(x_5_1 + x_3_1))  ## (BS, 64, ***, ***)
         
        x_mul  = torch.mul(x_3_2, x_5_2)
        
        out    = self.relu(x1 + self.conv_1_5(x_mul + x_3_1 + x_5_1))
         
        return out    
    
    
###################################################################
class FeaFusion(nn.Module):
    def __init__(self, channels):
        self.init__ = super(FeaFusion, self).__init__()
        
        self.relu     = nn.ReLU()
        self.layer1   = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        
        self.layer2_1 = nn.Conv2d(channels, channels //4, kernel_size=3, stride=1, padding=1)
        self.layer2_2 = nn.Conv2d(channels, channels //4, kernel_size=3, stride=1, padding=1)
        
        self.layer_fu = nn.Conv2d(channels//4, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2):
        
        ###
        wweight    = nn.Sigmoid()(self.layer1(x1+x2))
        
        ###
        xw_resid_1 = x1+ x1.mul(wweight)
        xw_resid_2 = x2+ x2.mul(wweight)
        
        ###
        x1_2       = self.layer2_1(xw_resid_1)
        x2_2       = self.layer2_2(xw_resid_2)
        
        out        = self.relu(self.layer_fu(x1_2 + x2_2))
        
        return out
    
###################################################################  
class FeaProp(nn.Module):
    def __init__(self, in_planes):
        self.init__ = super(FeaProp, self).__init__()
        

        act_fn = nn.ReLU(inplace=True)
        
        self.layer_1  = nn.Sequential(nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(in_planes),act_fn)
        self.layer_2  = nn.Sequential(nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(in_planes),act_fn)
        
        self.gate_1   = nn.Conv2d(in_planes*2, 1, kernel_size=1, bias=True)
        self.gate_2   = nn.Conv2d(in_planes*2, 1, kernel_size=1, bias=True)

        self.softmax  = nn.Softmax(dim=1)
        

    def forward(self, x10, x20):
        
        ###
        x1 = self.layer_1(x10)
        x2 = self.layer_2(x20)
        
        cat_fea = torch.cat([x1,x2], dim=1)
        
        ###
        att_vec_1  = self.gate_1(cat_fea)
        att_vec_2  = self.gate_2(cat_fea)

        att_vec_cat  = torch.cat([att_vec_1, att_vec_2], dim=1)
        att_vec_soft = self.softmax(att_vec_cat)
        
        att_soft_1, att_soft_2 = att_vec_soft[:, 0:1, :, :], att_vec_soft[:, 1:2, :, :]
        x_fusion = x1 * att_soft_1 + x2 * att_soft_2
        
        return x_fusion    
    
###################################################################      

class FAPNet(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(FAPNet, self).__init__()
        
        
        act_fn           = nn.ReLU(inplace=True)
        self.nf          = channel

        self.resnet      = res2net50_v1b_26w_4s(pretrained=True)
        self.downSample  = nn.MaxPool2d(2, stride=2)
        
        ##  
        self.rf1         = MFAM0(64,  self.nf)
        self.rf2         = MFAM(256,  self.nf)
        self.rf3         = MFAM(512,  self.nf)
        self.rf4         = MFAM(1024, self.nf)
        self.rf5         = MFAM(2048, self.nf)
        
        
        ##
        self.cfusion2    = FeaFusion(self.nf)
        self.cfusion3    = FeaFusion(self.nf)
        self.cfusion4    = FeaFusion(self.nf)
        self.cfusion5    = FeaFusion(self.nf)
        
        ##
        self.cgate5      = FeaProp(self.nf)
        self.cgate4      = FeaProp(self.nf)
        self.cgate3      = FeaProp(self.nf)
        self.cgate2      = FeaProp(self.nf)
        
        
        self.de_5        = nn.Sequential(nn.Conv2d(self.nf, self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn)
        self.de_4        = nn.Sequential(nn.Conv2d(self.nf, self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn)
        self.de_3        = nn.Sequential(nn.Conv2d(self.nf, self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn)
        self.de_2        = nn.Sequential(nn.Conv2d(self.nf, self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn)
        

        
        ##
        self.edge_conv0 = nn.Sequential(nn.Conv2d(64,       self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn) 
        self.edge_conv1 = nn.Sequential(nn.Conv2d(256,      self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn) 
        self.edge_conv2 = nn.Sequential(nn.Conv2d(self.nf,  self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn) 
        self.edge_conv3 = BasicConv2d(self.nf,   1,  kernel_size=3, padding=1)
        
        
        self.fu_5        = nn.Sequential(nn.Conv2d(self.nf*2, self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn)
        self.fu_4        = nn.Sequential(nn.Conv2d(self.nf*2, self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn)
        self.fu_3        = nn.Sequential(nn.Conv2d(self.nf*2, self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn)
        self.fu_2        = nn.Sequential(nn.Conv2d(self.nf*2, self.nf, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(self.nf),act_fn)
        
        
        ##
        self.layer_out5  = nn.Sequential(nn.Conv2d(self.nf, 1,  kernel_size=3, stride=1, padding=1))
        self.layer_out4  = nn.Sequential(nn.Conv2d(self.nf, 1,  kernel_size=3, stride=1, padding=1))
        self.layer_out3  = nn.Sequential(nn.Conv2d(self.nf, 1,  kernel_size=3, stride=1, padding=1))
        self.layer_out2  = nn.Sequential(nn.Conv2d(self.nf, 1,  kernel_size=3, stride=1, padding=1))
        
        
       
        ##
        self.up_2        = nn.Upsample(scale_factor=2,  mode='bilinear', align_corners=True)
        self.up_4        = nn.Upsample(scale_factor=4,  mode='bilinear', align_corners=True)
        self.up_8        = nn.Upsample(scale_factor=8,  mode='bilinear', align_corners=True)
        self.up_16       = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        


    def forward(self, xx):
        
        # ---- feature abstraction -----
        x   = self.resnet.conv1(xx)
        x   = self.resnet.bn1(x)
        x   = self.resnet.relu(x)
        
        # - low-level features
        x1  = self.resnet.maxpool(x)       # (BS, 64, 88, 88)
        x2  = self.resnet.layer1(x1)       # (BS, 256, 88, 88)
        x3  = self.resnet.layer2(x2)       # (BS, 512, 44, 44)
        x4  = self.resnet.layer3(x3)     # (BS, 1024, 22, 22)
        x5  = self.resnet.layer4(x4)     # (BS, 2048, 11, 11)
        
        ## -------------------------------------- ##
        xf1 = self.rf1(x1)
        xf2 = self.rf2(x2)
        xf3 = self.rf3(x3)
        xf4 = self.rf4(x4)
        xf5 = self.rf5(x5)
        
        
        ## edge 
        x21           = self.edge_conv1(x2)
        edge_guidance = self.edge_conv2(self.edge_conv0(x1) + x21)
        edge_out      = self.up_4(self.edge_conv3(edge_guidance))
        

        ### layer 5
        en_fusion5   = self.cfusion5(self.up_2(xf5), xf4)              ## (BS, 64, 22, 22)
        out_gate_fu5 = self.fu_5(torch.cat((en_fusion5, F.interpolate(edge_guidance, scale_factor=1/4, mode='bilinear')),dim=1))
        out5         = self.up_16(self.layer_out5(out_gate_fu5))
        
        
        de_feature4  = self.de_4(self.up_2(en_fusion5))                       ## (BS, 64, 22, 22)
        en_fusion4   = self.cfusion4(self.up_2(xf4), xf3)              ## (BS, 64, 44, 44)
        out_gate4    = self.cgate4(en_fusion4, de_feature4) ## (BS, 64, 44, 44) 
        out_gate_fu4 = self.fu_4(torch.cat((out_gate4, F.interpolate(edge_guidance, scale_factor=1/2, mode='bilinear')),dim=1))
        out4         = self.up_8(self.layer_out4(out_gate_fu4))
        
        
        de_feature3  = self.de_3(self.up_2(out_gate4))                 ## (BS, 64, 88, 88)
        en_fusion3   = self.cfusion3(self.up_2(xf3), xf2)              ## (BS, 64, 88, 88)
        out_gate3    = self.cgate3(en_fusion3, de_feature3)            ## (BS, 64, 88, 88)  
        out_gate_fu3 = self.fu_3(torch.cat((out_gate3, edge_guidance),dim=1))
        out3         = self.up_4(self.layer_out3(out_gate_fu3))
        
        
        de_feature2  = self.de_2(self.up_2(out_gate3))                 ## (BS, 64, 176, 176)
        en_fusion2   = self.cfusion2(self.up_2(xf2), self.up_2(xf1))   ## (BS, 64, 176, 176)
        out_gate2    = self.cgate2(en_fusion2, de_feature2)            ## (BS, 64, 176, 176)  
        out_gate_fu2 = self.fu_2(torch.cat((out_gate2, self.up_2(edge_guidance)), dim=1))
        out2         = self.up_2(self.layer_out2(out_gate_fu2))

        
        # ---- output ----
        return out5, out4, out3, out2, edge_out
    
     
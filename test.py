import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse


from lib.model import FAPNet
from utils.dataloader import tt_dataset
from utils.eva_funcs import eval_Smeasure,eval_mae,numpy2tensor
import scipy.io as scio 
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--testsize',   type=int, default=352, help='the snapshot input size')
parser.add_argument('--model_path', type=str, default='./checkpoints/')
parser.add_argument('--save_path',  type=str, default='./results/')

opt   = parser.parse_args()
model = FAPNet(channel=64).cuda()


cur_model_path = opt.model_path+'FAPNet.pth'   
model.load_state_dict(torch.load(cur_model_path))
model.eval()
        
    
################################################################

for dataset in ['CHAMELEON', 'CAMO', 'COD10K']:
    
    save_path = opt.save_path + dataset + '/'
    os.makedirs(save_path, exist_ok=True)        
        
        
    test_loader = tt_dataset('/test/CamouflagedObjectDection/Dataset/TestDataset/{}/Imgs/'.format(dataset),
                               '/test/CamouflagedObjectDection/Dataset/TestDataset/{}/GT/'.format(dataset), opt.testsize)
        

    
    for iteration in range(test_loader.size):
            
            
        image, gt, name = test_loader.load_data()

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
            

        _,_,_, cam,_ = model(image)
            
        res = F.upsample(cam, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        
################################################################
        cv2.imwrite(save_path+name, res*255)
        


 
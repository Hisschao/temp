import sys
sys.path.append("../")
import os
import numpy as np
import random
import torch
import time
from torch.utils.data.dataset import Dataset
from osgeo import gdal, gdalconst, osr
import glob

import MODEL_nomasking as MODEL

model_path = 'MM_VSF.pt'
device = 'cuda:0'
bsize = 4
t = 6
data_sat = torch.randn(bsize, t, 6, 128, 128)
data_weather = torch.randn(bsize,365,5)
timestamps = torch.randint(3, 300, (bsize,t+1)).float()
delt_timestamps = torch.randint(3, 20, (bsize,t)).float()

# choose model you want from Model_nomasking.py
model = MODEL.MM_VSF(image_size = 128, patch_size = 8, num_classes = 1000, dim = 512, channels = 6, depth = 4, heads = 8, mlp_dim = 1024, pool = 'cls', decoder_dim = 512, masking_ratio = 0.75, decoder_depth = 1, decoder_heads = 8, decoder_dim_head = 64, doy_embed_dim = 16, delt_embed_dim = 16, channels_w_in = 5,channels_w_out = 5, dim_w = 32)
model = model.to(device)

model.load_state_dict(torch.load(os.path.join(model_path),map_location=device))
model.eval()

out_f,mask,embs = model(data_sat.to(device),data_weather.to(device),timestamps.to(device),delt_timestamps.to(device))
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("MM_VSF")
print("Total parameters:",pytorch_total_params)
print("Input Data Shapes: ",data_sat.shape, data_weather.shape)
print("Output shape: ",out_f.shape)
print("Embeddings shape",embs.shape)
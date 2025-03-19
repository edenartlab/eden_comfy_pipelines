import cv2
import torch
import numpy as np
from PIL import Image

def preprocess_lab(lab):
    L_chan, a_chan, b_chan = torch.unbind(lab, dim=2)
    # L_chan: black and white with input range [0, 100]
    # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
    # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
    return [L_chan / 50.0 - 1.0, a_chan / 110.0, b_chan / 110.0]


def deprocess_lab(L_chan, a_chan, b_chan):
    # TODO This is axis=3 instead of axis=2 when deprocessing batch of images 
    # (we process individual images but deprocess batches)
    return torch.stack([(L_chan + 1) / 2.0 * 100.0, a_chan * 110.0, b_chan * 110.0], dim=2)


def rgb_to_lab(srgb):
    # Get appropriate device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    srgb_pixels = torch.reshape(srgb, [-1, 3])
    
    # Convert to correct device
    srgb_pixels = srgb_pixels.to(device)

    linear_mask = (srgb_pixels <= 0.04045).type(torch.FloatTensor).to(device)
    exponential_mask = (srgb_pixels > 0.04045).type(torch.FloatTensor).to(device)
    rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
    
    rgb_to_xyz = torch.tensor([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ]).type(torch.FloatTensor).to(device)
    
    xyz_pixels = torch.mm(rgb_pixels, rgb_to_xyz)
    
    # XYZ to Lab
    xyz_normalized_pixels = torch.mul(xyz_pixels, torch.tensor([1/0.950456, 1.0, 1/1.088754]).type(torch.FloatTensor).to(device))

    epsilon = 6.0/29.0

    linear_mask = (xyz_normalized_pixels <= (epsilon**3)).type(torch.FloatTensor).to(device)
    exponential_mask = (xyz_normalized_pixels > (epsilon**3)).type(torch.FloatTensor).to(device)

    fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4.0/29.0) * linear_mask + ((xyz_normalized_pixels+0.000001) ** (1.0/3.0)) * exponential_mask
    
    # Convert to lab
    fxfyfz_to_lab = torch.tensor([
        #  l       a       b
        [  0.0,  500.0,    0.0], # fx
        [116.0, -500.0,  200.0], # fy
        [  0.0,    0.0, -200.0], # fz
    ]).type(torch.FloatTensor).to(device)
    
    lab_pixels = torch.mm(fxfyfz_pixels, fxfyfz_to_lab) + torch.tensor([-16.0, 0.0, 0.0]).type(torch.FloatTensor).to(device)
    
    return torch.reshape(lab_pixels, srgb.shape)


def lab_to_rgb(lab):
    # Get appropriate device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    lab_pixels = torch.reshape(lab, [-1, 3])
    
    # Convert to device
    lab_pixels = lab_pixels.to(device)
    
    # Convert to fxfyfz
    lab_to_fxfyfz = torch.tensor([
        #   fx      fy        fz
        [1/116.0, 1/116.0,  1/116.0], # l
        [1/500.0,     0.0,      0.0], # a
        [    0.0,     0.0, -1/200.0], # b
    ]).type(torch.FloatTensor).to(device)
    
    fxfyfz_pixels = torch.mm(lab_pixels + torch.tensor([16.0, 0.0, 0.0]).type(torch.FloatTensor).to(device), lab_to_fxfyfz)

    # Convert to xyz
    epsilon = 6.0/29.0
    linear_mask = (fxfyfz_pixels <= epsilon).type(torch.FloatTensor).to(device)
    exponential_mask = (fxfyfz_pixels > epsilon).type(torch.FloatTensor).to(device)

    xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29.0)) * linear_mask + ((fxfyfz_pixels+0.000001) ** 3) * exponential_mask

    # Denormalize for D65 white point
    xyz_pixels = torch.mul(xyz_pixels, torch.tensor([0.950456, 1.0, 1.088754]).type(torch.FloatTensor).to(device))

    xyz_to_rgb = torch.tensor([
        #     r           g          b
        [ 3.2404542, -0.9692660,  0.0556434], # x
        [-1.5371385,  1.8760108, -0.2040259], # y
        [-0.4985314,  0.0415560,  1.0572252], # z
    ]).type(torch.FloatTensor).to(device)

    rgb_pixels = torch.mm(xyz_pixels, xyz_to_rgb)
    
    # Avoid a slightly negative number messing up the conversion
    # Clip
    rgb_pixels = torch.clamp(rgb_pixels, 0.0, 1.0)

    linear_mask = (rgb_pixels <= 0.0031308).type(torch.FloatTensor).to(device)
    exponential_mask = (rgb_pixels > 0.0031308).type(torch.FloatTensor).to(device)
    srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + (((rgb_pixels+0.000001) ** (1/2.4) * 1.055) - 0.055) * exponential_mask
    
    # Move back to original device if needed
    if lab.device != device:
        srgb_pixels = srgb_pixels.to(lab.device)
        
    return torch.reshape(srgb_pixels, lab.shape)
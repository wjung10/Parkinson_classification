import os
import sys
from threading import Thread
import einops
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import torch
import segmentation_models_pytorch as smp
from numpy.linalg import inv
from scipy.ndimage import affine_transform
import time
import copy

list_dir_model = [
    './models/axial.pt',
    './models/coronal.pt',
    './models/sagittal.pt',
]

parts = ['Midbrain', 'Pons', 'Medulla', 'SCP']

list_models = []
for i in range(len(list_dir_model)):
    model = smp.Unet('resnet34', decoder_channels=[128,64,32,16,8], classes=5, encoder_weights=None)
    model.load_state_dict(torch.load(list_dir_model[i]))
    model.requires_grad_(False)
    model.eval().cuda()
    list_models.append(model)

def zoom_bs(x, y, z=0.5):
    assert type(x) == nib.Nifti1Image
    assert type(y) == nib.Nifti1Image
    x_data = x.get_data()
    y_data = y.get_data()
    bs = np.where(y_data == 101, 1, 0)
    coords = np.argwhere(bs > 0)
    cm = coords[:,0].mean(), coords[:,1].mean(), coords[:,2].mean()
    
    Mdc = [[-1, 0, 0], [0, 0, -1], [0, 1, 0]]
    delta = z
    MdcD = np.array(Mdc).T * delta
    
    vol_center = MdcD.dot(np.array([256, 256, 256]) / 2)
    Pxyz_c = y.affine.dot(np.hstack((cm, [1])))[:3]
    out_affine = nib.affines.from_matvec(MdcD, Pxyz_c - vol_center)
   
    vox2vox = inv(out_affine) @ y.affine
    
    y_mapped_data = affine_transform(y_data, inv(vox2vox), output_shape=[256, 256, 256], order=0, mode='constant', cval=0)
    x_mapped_data = affine_transform(x_data, inv(vox2vox), output_shape=[256, 256, 256], order=0, mode='constant', cval=0)
    return nib.Nifti1Image(x_mapped_data, out_affine), nib.Nifti1Image(y_mapped_data, out_affine), vox2vox 

def run_model(x, model, view, probability_map=None, return_pred=False):
    if probability_map == None:
        probability_map = torch.zeros((5, 256, 256, 256)).cuda()
        probability_map[0,:,:,0] = 1
        probability_map[0,:,:,255] = 1
    x = x.float()
    if not x.is_cuda:
        x = x.cuda()
    slices = [slice(0, 256, 1)] * 3
    slice_dim = {'axial': 1, 'sagittal': 0, 'coronal': 2}[view]
    transpose_ = {'axial': 'w c h -> c w h', 'coronal': 'w h c -> c w h', 'sagittal': 'c w h -> c w h'}[view]
    for i in range(1, 255):
        slices[slice_dim] = slice(i-1, i+2, 1)
        x_slice = einops.rearrange(x[tuple(slices)], transpose_)
        x_slice = x_slice.unsqueeze(0)
        if x_slice.max() > 0:
            x_slice = x_slice / x_slice.max()
        out = model(x_slice)
        out = F.softmax(out, dim=1)[0]
        
        slices[slice_dim] = i
        probability_map[tuple([slice(0, 5, 1)] + slices)] += out
    if return_pred:
        return probability_map.argmax(axis=0).cpu()

def run_ensemble(x, axial_model, coronal_model, sagittal_model):
    x = x.cuda().float()
    probability_map = torch.zeros((5, 256, 256, 256)).cuda()
    probability_map[0,:,:,0] = 1
    probability_map[0,:,:,255] = 1
    
    th1 = Thread(target=run_model, args=(x, axial_model, 'axial', probability_map))
    th2 = Thread(target=run_model, args=(x, coronal_model, 'coronal', probability_map))
    th3 = Thread(target=run_model, args=(x, sagittal_model, 'sagittal', probability_map))
    th1.start()
    th2.start()
    th3.start()
    th1.join()
    th2.join()
    th3.join()
    pred = probability_map.argmax(axis=0).cpu()
    return pred

def run_brainstem_seg(t1_path, seg_path, save_path):
    prep_t1 = nib.load(t1_path)
    db_output = nib.load(seg_path) 
    
    zx, zy, vox2vox = zoom_bs(prep_t1, db_output)
    zpred = run_ensemble(torch.tensor(zx.get_fdata()), list_models[0], list_models[1], list_models[2]).numpy()
    pred = affine_transform(zpred, vox2vox, output_shape=[256, 256, 256], order=0)
    
    volumes = [0, 0, 0, 0]
    indices, counts = np.unique(pred, return_counts=True)
    
    nib.save(nib.Nifti1Image(pred.astype(np.uint8), db_output.affine), save_path)

if __name__ == '__main__':
    t1_path = sys.argv[1]
    seg_path = sys.argv[2]
    save_path = sys.argv[3]
    run_brainstem_seg(t1_path, seg_path, save_path)



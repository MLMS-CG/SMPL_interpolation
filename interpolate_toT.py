# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 17:46:57 2024

@author: Boyang YU
"""

import os
#import os.path as osp
import numpy as np
#from global_var import ROOT

from smpl_torch import SMPLNP_Lres
import trimesh
import torch
from smpl_interpolation import interpolate_pose
import joblib

from tqdm import tqdm

def get_smpl(theta, beta, gender="male"):
    #gender = 'male'

    smpl = SMPLNP_Lres(gender=gender, cuda=False)
    body_f = smpl.base.faces

    theta=theta.reshape(1,72)
    beta=beta.reshape(1,10)

    body_v = smpl(beta,theta)
    body_trimesh = trimesh.Trimesh(body_v, body_f, process=False)
    return body_trimesh

def retrive_sequence(shape,thetas, gender="male", folder_path="./fat_walking"):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully")
    else:
        print(f"Folder '{folder_path}' already exists")
    
    for i in tqdm(range(len(thetas))):
        pose=thetas[i]
        body_trimesh=get_smpl(pose, shape, gender)

        new_body_data=trimesh.exchange.obj.export_obj(body_trimesh)
        with open(folder_path+"/body_{}.obj".format(i),"w") as f:
                f.write(new_body_data)

def quat_axisangle(quat):
    #batch_size = quat.shape[0]

    # Extracting the scalar and vector parts of the quaternion
    q_scalar = quat[:, : ,0:1]
    q_vector = quat[:, : ,1:]
    
    #print(q_scalar.shape)
    #print(q_vector.shape)

    # Calculating the angle using arccos of the scalar part
    angle =  2*torch.acos(q_scalar)
    #rint(angle)
    
    # Normalizing the quaternion vector part
    #q_normalized = q_vector / torch.norm(q_vector + 1e-8, p=2, dim=1, keepdim=True)
    q_normalized = q_vector / (torch.sin(torch.acos(q_scalar))+1e-8)
    
    # Obtaining the axis-angle representation
    axisangle = q_normalized * angle  # Broadcasting the angle to each row

    return axisangle

if __name__ == '__main__':
    
    
    dict_ = joblib.load('./samples_test.pt')
    print(dict_.keys())
    seq_name="0_arms_move"
    """
    You can replace with any pose sequence and shape sequence
    """
    shape=dict_[seq_name][0,72:]
    poses=dict_[seq_name][:,:72]
    initial_pose= dict_[seq_name][0,:72].reshape(-1,3) #24,3
    
    
    # generate the meshes of the given sequence
    retrive_sequence(shape, poses, folder_path=seq_name)
    
    # interpolate between canonical pose and the first pose of the sequence 
    pose0= np.zeros((24,3)) # all zeros, canonical pose
    pose0[0]=initial_pose[0]

    
    ts=[0.05*i for i in range(1,20)]  # here you set the intervals of interpolation
    slerp_poses=interpolate_pose(pose0, initial_pose, ts)
    
    inbetween_poses=quat_axisangle(slerp_poses)
    retrive_sequence(shape, inbetween_poses.reshape(len(ts),-1), folder_path=seq_name+"_prepend") # generate the interpolated meshes from T to the first frame. 
    
    





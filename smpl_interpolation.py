# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 20:57:56 2023

@author: Boyang
"""

import torch
import numpy as np
#from SMPL import getSMPL


def axisangle_quat(theta):  
    #theta N x 3 24*3
    #batch_size = theta.shape[0]
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    #print("quat shape: ", quat.shape)
    return quat

def slerp_new(starting_q, ending_q, t ):

  #Scalar d = this->dot(other);
  d=torch.sum(starting_q*ending_q,axis=1,keepdims=True) #24,1
  absD=torch.abs(d)
  

  scale0=torch.zeros_like(absD)
  scale1=torch.zeros_like(absD)
  

  # by default
  theta = torch.acos(absD);
  sinTheta = torch.sin(theta);
  scale0 = torch.sin( ( 1 - t ) * theta) / sinTheta;
  scale1 = torch.sin( ( t * theta) ) / sinTheta;
  
   
  scale0=torch.where(absD>=1-1e-5,1-t,scale0)
  scale1=torch.where(absD>=1-1e-5,t,scale1)
  
  
  
  scale1=torch.where(d<0, -scale1, scale1)


  #if(d<Scalar(0)) scale1 = -scale1;

  quat_res=scale0*starting_q + scale1*ending_q
  
  return quat_res
  #return Quaternion<Scalar>(scale0 * coeffs() + scale1 * other.coeffs());

def interpolate_pose(pose0,pose1,ts):  
    print(pose0.shape)
    q0=axisangle_quat(torch.from_numpy(pose0)) #24,3->24,4
    q1=axisangle_quat(torch.from_numpy(pose1)) 
    results=[]
    
    for t in ts:
        res=slerp_new(q0,q1,t)
        #res=slerp_batch(q0, q1, t)
        results.append(res)
    
    print("interpolated",torch.stack(results).shape)
    return torch.stack(results) #n,24,4

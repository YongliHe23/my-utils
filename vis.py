
import matplotlib.pyplot as plt
from typing import Optional
import torch
import numpy as np

def SACshow(img,
            ix: Optional[int]=None,
            iy: Optional[int]=None,
            iz: Optional[int]=None, *, 
            cmap='gray'):
    r'''Visualize a 3D image
    Input:
    - ``img``: 3d array to be viewed
    Optional:
    - ``ix/iy/iz``: the slice(s) to be viewed along x/y/z axis
    - ``cmap``: color map
    
    '''
    
    nx,ny,nz=img.shape
    
    if ix==None:
        ix=nx//2 #show mid plane
    if iy==None:
        iy=ny//2
    if iz==None:
        iz=nz//2
    assert (ix>0 & ix<nx-1), 'x index must be positive integer and not out of bound'
    assert (iy>0 & iy<ny-1), 'y index must be positive integer and not out of bound'
    assert (iz>0 & iz<nz-1), 'z index must be positive integer and not out of bound'

    slc1=img[ix,:,:]
    slc2=img[:,iy,:]
    slc3=img[:,:,iz]
    slc=list([slc1,slc2,slc3])

    fig,ax=plt.subplots(1,3, figsize=(20,10))

    i=0
    for axis in ax.flat:
        im=axis.imshow(slc[i],cmap=cmap)
        i+=1
    return

def pulse2np(pInit):
    gx=pInit.gr[0,0,:].cpu().numpy()
    #gx.shape
    gy=pInit.gr[0,1,:].cpu().numpy()
    gz=pInit.gr[0,2,:].cpu().numpy()

    #pInit.rf.shape
    b1x=pInit.rf[0,0,:]
    b1y=pInit.rf[0,1,:]
    rf_mag=(b1x**2+b1y**2)**0.5
    rf_mag=rf_mag.cpu().numpy()

    rf_phase=torch.arctan(b1y/b1x).cpu().numpy()
    return gx,gy,gz,rf_mag,rf_phase

def plot_seq(gx,gy,gz,rf_mag,rf_phase):
    tt=np.arange(0,gx.shape[-1])
    fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(8,10))
    ax1.plot(tt,gx,label='x')
    ax1.plot(tt,gy,label='y')
    ax1.plot(tt,gz,label='z')

    ax1.set_title('Gr')
    ax1.legend()

    ax2.plot(tt,rf_mag)
    ax2.set_title('|b1|')

    ax3.plot(tt,rf_phase)
    ax3.set_title('∠b1')

    plt.tight_layout()
    plt.show()
    return

def g2k(g,dt=4e-3,doPlot=True):
    '''
    Inputs:
     - g: (3,nT)
     - dt: raster time (ms)
     - doPlot: True/false
    '''
    k=dt*4.2576*torch.cumsum(torch.flip(g,dims=[1]),axis=1)
    k=torch.flip(k,dims=[1]) #(3,nT)
    k=k.T#(nT,3)
    
    kx=k[:,0].numpy()
    ky=k[:,1].numpy()
    kz=k[:,2].numpy()
    
    print(kx.shape, ky.shape, kz.shape)
    
    if doPlot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(kx,ky,kz)
        ax.set_xlabel('kx')
        ax.set_ylabel('ky')
        ax.set_zlabel('kz')
        plt.show()
    return k

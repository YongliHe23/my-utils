
import matplotlib.pyplot as plt
from typing import Optional
import torch
import numpy as np

def SACshow(img,
            ix: Optional[int]=None,
            iy: Optional[int]=None,
            iz: Optional[int]=None, *, 
            transpose=True,
            pad= True,
            cmap='gray',
            caxis= None):
    r'''Visualize a 3D image
    Input:
    - ``img``: 3d array to be viewed
    Optional:
    - ``ix/iy/iz``: the slice(s) to be viewed along x/y/z axis
    - ``transpose``: transpose the array when plt.imshow()
    - ``pad``: pad the array with 0 to be a cubic array
    - ``cmap``: color map
    - ``caxis``: color limits, don't show color bar and use default color limits if None
    
    '''
    if (img.device!='cpu'):
        img=img.cpu()
    nx,ny,nz=img.shape
    
    nmax=np.max(np.array([nx,ny,nz]))
    if pad:
        pimg=np.pad(img,pad_width=(((nmax-nx)//2,(nmax-nx)//2),((nmax-ny)//2,(nmax-ny)//2),((nmax-nz)//2,(nmax-nz)//2)),mode='constant', constant_values=0)
    else:
        pimg=img
        
    nnx,nny,nnz=pimg.shape #new nx/ny/nz
        
    if ix==None:
        ix=nnx//2 #show mid plane
    if iy==None:
        iy=nny//2
    if iz==None:
        iz=nnz//2
    assert (ix>0 & ix<nnx-1), 'x index must be positive integer and not out of bound'
    assert (iy>0 & iy<nny-1), 'y index must be positive integer and not out of bound'
    assert (iz>0 & iz<nnz-1), 'z index must be positive integer and not out of bound'

    slc1=pimg[ix,:,:]
    slc2=pimg[:,iy,:]
    slc3=pimg[:,:,iz]
    slc=list([slc1.T,slc2.T,slc3.T])
    title=['sag.','coro.','ax.']
    
    fig,ax=plt.subplots(1,3, figsize=(20,10))

    i=0
    for axis in ax.flat:
        im=(axis.imshow(slc[i],origin='lower', cmap=cmap) if caxis is None 
           else axis.imshow(slc[i],origin='lower',cmap=cmap,vmin=caxis[0],vmax=caxis[1]))
        axis.set_title(title[i])
        i+=1
        
    if caxis is not None:
        cbar = fig.colorbar(im, ax=ax, location='right', shrink=0.6)
        cbar.set_label("Color Intensity")

        
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

def plot_seq(pulse):
    r''' display mobjs.Pulse object
    '''
    [gx,gy,gz,rf_mag,rf_phase]=pulse2np(pulse)
    
    tt=np.arange(0,gx.shape[-1])*(pulse.dt.numpy() if pulse.device=='cpu' else pulse.dt.cpu().numpy())*1e3
    
    fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(8,10))
    ax1.plot(tt,gx,label='x')
    ax1.plot(tt,gy,label='y')
    ax1.plot(tt,gz,label='z')
    
    ax1.set_xlabel('time(ms)')
    ax1.set_ylabel('Gauss/cm')
    ax1.set_title('Gr')
    ax1.legend()

    ax2.plot(tt,rf_mag)
    ax2.set_ylabel('Gauss')
    ax2.set_title('|b1|')

    ax3.plot(tt,rf_phase)
    ax3.set_ylabel('Rad')
    ax3.set_title('∠b1')

    plt.tight_layout()
    plt.show()
    return

def compare_pulses(pulses,legends):
    r'''display multiple mobjs.Pulse objects for comparison
    Input:
    - ``pulses``: list of Pulse objects
    - ``legends``: list of legends
    '''
    dt=pulses[0].dt.numpy() if pulses[0].device=='cpu' else pulses[0].dt.cpu().numpy()
    
    tt=np.arange(0,pulses[0].gr.shape[-1])*(dt)*1e3
    fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,1,figsize=(8,10))
    
    
    ax1.set_ylabel('Gx(G/cm)')
    #ax1.set_title('Gx')
    ax2.set_ylabel('Gy(G/cm)')
    #ax2.set_title('Gy')
    ax3.set_ylabel('Gz/(G/cm)')
    #ax3.set_title('Gz')
    
    ax4.set_ylabel('|b1|(Gauss)')
    #ax4.set_title('|b1|')
    
    ax5.set_ylabel('∠b1(Rad)')
    #ax5.set_title('∠b1')
    ax5.set_xlabel('time(ms)')
    
    for i,p in enumerate(pulses):
        [gx,gy,gz,rf_mag,rf_phase]=pulse2np(p)
        ax1.plot(tt,gx,label=legends[i])
        ax2.plot(tt,gy,label=legends[i])
        ax3.plot(tt,gz,label=legends[i])
        ax4.plot(tt,rf_mag,label=legends[i])
        ax5.plot(tt,rf_phase,label=legends[i])
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    
    plt.tight_layout()
    plt.show()
    return
    

def g2k(grads,legends=None, dt=4e-3,doPlot=True):
    '''
    Inputs:
     - grads: list of gradient arrays each g=(3,nT)
     - legends: list of legend(s)
     - dt: raster time (ms)
     - doPlot: True/false
    '''
    ktraj=[]
    for i,g in enumerate(grads):
        k=dt*4.2576*torch.cumsum(torch.flip(g,dims=[1]),axis=1)
        k=torch.flip(k,dims=[1]) #(3,nT)
        k=k.T#(nT,3)
        ktraj.append(k)
    
    #print(kx.shape, ky.shape, kz.shape)
    
    if doPlot:
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111, projection='3d')
        for i,k in enumerate(ktraj):
            kx=k[:,0].numpy()
            ky=k[:,1].numpy()
            kz=k[:,2].numpy()
            if legends is not None:
                ax.plot(kx,ky,kz,label=legends[i])
            else:
                ax.plot(kx,ky,kz)
        ax.set_xlabel('kx(1/cm)')
        ax.set_ylabel('ky(1/cm)')
        ax.set_zlabel('kz(1/cm)',labelpad=-0.5)
        if legends is not None:
            ax.legend()
        plt.tight_layout()
        plt.show()
    return ktraj

def errormap_cmp(images,titles,caxis,cmap='jet'):
    r''' show image1-3 in same colorbar
    images: list of images to be displayed
    titles: titles for each subplot
    caxis: color limits for the shared cbar, list of 2 values
    cmap: color map
    '''
    n=len(images)
    
    # Create figure and set up gridspec for proper colorbar alignment
    fig = plt.figure(figsize=(4*n, 4))
    gs = fig.add_gridspec(1, n+1,width_ratios=[1] * n + [0.05])  # n images + 1 colorbar
    
    
    axes=[]
    imgs = []
    for i,image in enumerate(images):
        ax = fig.add_subplot(gs[0, i])
        img = ax.imshow(image, cmap=cmap, vmin=caxis[0], vmax=caxis[1])
        ax.set_title(titles[i])
       # ax.axis('off')  # Remove axis labels
        axes.append(ax)  # Store the axis
        imgs.append(img)  # Store the image

    # Add a shared colorbar
    cbar = fig.colorbar(imgs[-1], cax=fig.add_subplot(gs[0, i+1]))
    cbar.set_label("Color Intensity")

    plt.tight_layout()
    plt.show()
    return

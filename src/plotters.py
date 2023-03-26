import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def fig2img ( fig ):
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )


def plot_y(Grey, Y, Y_ema, Color):

    imgs = torch.cat([Grey, Y, Y_ema, Color]).permute(0,2,3,1).mul(0.5).add(0.5).numpy().clip(0,1)

    fig, axes = plt.subplots(4, 8, figsize=(15, 4.5), dpi=150)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
        
    axes[0, 0].set_ylabel('Grey', fontsize=24)
    axes[1, 0].set_ylabel('Y', fontsize=24)
    axes[2, 0].set_ylabel('Y_ema', fontsize=24)
    axes[3, 0].set_ylabel('Color', fontsize=24)
    
    fig.tight_layout(pad=0.001)
    return fig, axes


def plot_noise(ref, pred):
    imgs = torch.cat([ref[:10], pred[:10]]).permute(0,2,3,1).mul(0.5).add(0.5).numpy().clip(0,1)

    fig, axes = plt.subplots(2, 10, figsize=(15, 4.5), dpi=150)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
        
    axes[0, 0].set_ylabel('Reference', fontsize=24)
    axes[1, 0].set_ylabel('Prediction', fontsize=24)
    
    fig.tight_layout(pad=0.001)
    return fig, axes

def print_stat(st):
    print(f"Min: {st.min()}, Max: {st.max()}, Mean: {st.mean()}, Std: {st.std()}")
#!env python
# -*- coding: utf-8 -*-
# Anisotropic diffusion of 2D image.
# implemented by t-suzuki.
# License: Public Domain
#
# reference:
# Pietro Perona, et al. "Scale-Space and Edge Detection Using Anisotropic Diffusion", IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, VOL. 12, NO.7, JULY 1990.

import sys
import skimage.color
import skimage.data
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt

def anisotropic_diffusion(img, N=10, K=-1.0, diffuse_function='exp', verbose=True):
    u'''calc anisotropic diffusion of image <img> (4 neighbor)'''
    auto_K = K < 0
    # determine g(||dI(t)||)
    def g_exp(v): return np.exp(-(v/K)**2.0)
    def g_rcp(v): return 1.0/(1.0 + (v/K)**2.0)
    g_func = {'exp': g_exp, 'rcp': g_rcp}[diffuse_function]

    def gradient(m):
        gradx = np.hstack([m[:, 1:], m[:, -2:-1]]) - np.hstack([m[:, 0:1], m[:, :-1]])
        grady = np.vstack([m[1:, :], m[-2:-1, :]]) - np.vstack([m[0:1, :], m[:-1, :]])
        return gradx, grady

    def W_diff(m): return np.hstack([m[:, 0:1], m[:, :-1]])   - m
    def E_diff(m): return np.hstack([m[:, 1:],  m[:, -2:-1]]) - m
    def N_diff(m): return np.vstack([m[0:1, :], m[:-1, :]])   - m
    def S_diff(m): return np.vstack([m[1:, :],  m[-2:-1, :]]) - m

    # iteration
    mu = 1.0/4.0
    I = np.array(img)
    for i in range(N):
        if auto_K:
            gx, gy = gradient(I)
            K = np.sqrt(gx**2.0 + gy**2.0).mean()*0.9 # constatn 0.9 from the paper.
        W, E, N, S = W_diff(I), E_diff(I), N_diff(I), S_diff(I)
        I_delta = W*g_func(np.abs(W)) + E*g_func(np.abs(E)) + N*g_func(np.abs(N)) + S*g_func(np.abs(S))
        I += mu*I_delta
        if verbose:
            sys.stdout.write('.')
            sys.stdout.flush()
    if verbose:
        sys.stdout.write('\n')
    return I

if __name__=='__main__':
    # load image and preprocess
    if len(sys.argv) > 1:
        fn = sys.argv[1]
        img = skimage.color.rgb2gray(skimage.data.imread(fn))
    else:
        img = skimage.color.rgb2gray(skimage.data.lena())
        img = skimage.transform.resize(img, (256, 256))
    img = img.astype(np.float32)

    # calc anisotropic diffusion
    N = 30
    K = 0.1
    filtered = anisotropic_diffusion(img, K=K, N=N)
    vmin, vmax = img.min(), img.max()

    # plot the results
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Anisotropic Diffusion (2D) Demo')
    ax = axs[0]; ax.imshow(img, cmap='gray'); ax.set_title('org')
    ax = axs[1]; ax.imshow(filtered, cmap='gray'); ax.set_title('anisotropic diffusion (%d)' % N)
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)

    # plot the results
    fig, axs = plt.subplots(2, 6, figsize=(24, 8))
    fig.suptitle('Anisotropic Diffusion (2D) Demo')
    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            if j == 0:
                # leftmost image
                ax.imshow(img, cmap='gray')
                ax.set_title('org')
            else:
                if i == 0: # top row: change the number of iteration
                    n_iter = j*10
                    K = -1
                    ax.set_title('K=auto %d iter' % n_iter)
                else: # bottom row: change K
                    n_iter = 30
                    K = 0.05*j
                    ax.set_title('K=%.3f %d iter' % (K, n_iter))
                filtered = anisotropic_diffusion(img, K=K, N=n_iter)
                ax.imshow(filtered, vmin=vmin, vmax=vmax, cmap='gray')
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    plt.show()


# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 13:35:44 2017

@author: tlouf

This module gives some essential functions to extract the data from a fits file,
build a matrix representing the image of each event and plotting it (showing it
in the console and/or saving it in a folder in Images/ attributed to the batch)
"""

from csv_manips import read_csv
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

seuils = read_csv('seuils')
reverse_gain = 0.22792108


def data_extract(file):
    ''' This function extract the lists of the pixel and frame number and of the
    amplitude of this pixel from the path to a fits file (variable "file"). The
    astropy.io.fits library makes it very convenient and easy to do, the extraction
    of the data working in a way very similar to how this is done with a dictionary.
    '''
    fits.info(file)
    hdulist = fits.open(file)
    data = hdulist[1].data
    hdulist.close()
    return(data.PIXEL, data.NFRAME, data.AMPLITUDE)


def img_construct(k, Levent, pixNs, amps):
    ''' This function builds a 16x16 matrix representing an event. It takes in
    argument the lists of pixel numbers and their amplitudes ("pixNs" and "amps"),
    which are output by "data_extract". It also has Levent, which is the list of
    the indices at which a new event starts, in both "pixNs" and "amps" (a given
    index corresponding to the same information). Finally, "k" designates the
    event number we are currently considering.
    '''
    image = np.zeros((16, 16))

    for i in range (Levent[k], Levent[k+1]):
        # The pixel numbers go from 0 to 255, as though you flattened a 16x16
        # matrix by concatenatitng each column to form a 1D array.
        xn = int(pixNs[i]%16)
        yn = 15 - int(pixNs[i]/16)
        image[yn, xn] = amps[i]

    return image


def img_construct_seuils(k, Levent, pixNs, amps):
    image = np.zeros((16, 16))

    for i in range (Levent[k], Levent[k+1]):
        xn = int(pixNs[i]%16)
        yn = 15 - int(pixNs[i]/16)
        image[yn, xn] = (amps[i] - seuils[int(pixNs[i])]) * reverse_gain

    return image


def events_browsing(batch):
    ''' This function is a generator which is used in numerous 'for' loops
    in phi_distrib and manip1 modules and in the visualization function here.
    It browses every file in a batch, and each event of each file, in order to
    yield the image matrix reprensenting it, as well as the file and event
    numbers (in order to save the images under a unique and sound name).
    '''
    fits_files = glob.glob('FITS/'+batch+'/*.fits')
    # 'glob' output is a list of files of the batch we're interested in, but
    # it orders the files suchs as 10 is before 2 (it prioritizes the first
    # digit to sort), hence the sorting in the for loop to iterate the same
    # way as we see the files in the batch folder.
    for file_nb, file in enumerate(sorted(fits_files, key=lambda name: int(name[6+len(batch):-5]))):
        pixNs, frameNs, amps = data_extract(file)
        n = len(frameNs)
        # We create a copy of the list of frame numbers, with a 0 added at the
        # start and the last element removed. We then produce Levent, the list
        # of the  indices at which an event start in frame_Ns, by determining
        # the indices at which the difference between each element of frame_Ns
        # and each element of its copy is equal to 0. Thus, adding 0 at the
        # start of the  copy is purely arbitrary, and is just to ensure that
        # the index 0 will be in Levent. Removing the last element ensures
        # that the two arrays are of same length and thus can be subtracted from
        # one another.
        frameNs_2 = np.delete(np.insert(frameNs, 0, 0), n)
        Levent = (frameNs_2-frameNs).nonzero()[0]
        # We then append n because of the way 'img_construct' works, this will
        # prevent the loop from forgetting the last event.
        np.append(Levent, n)
        for k in range (len(Levent)-1):
            yield (img_construct(k, Levent, pixNs, amps), file_nb, k)


def events_browsing_seuils(batch):
    fits_files = glob.glob('FITS/'+batch+'/*.fits')
    for file_nb, file in enumerate(sorted(fits_files, key=lambda name: int(name[6+len(batch):-5]))):
        pixNs, frameNs, amps = data_extract(file)
        n = len(frameNs)
        frameNs_2 = np.delete(np.insert(frameNs, 0, 0), n)
        Levent = (frameNs_2-frameNs).nonzero()[0]
        np.append(Levent, n)
        for k in range (len(Levent)-1):
            yield (img_construct_seuils(k, Levent, pixNs, amps), file_nb, k)


def heatmap(img, center, name='test', save=False, show=True, fig_scale=1, fig_dpi=None):
    ''' This function plots a heatmap of a single event from the matrix "img"
    built by "img_construct", which represents the amplitude of each pixel. It
    includes a  grid to better visualize the position of the pixels, proper
    axes and tick labels, a colorbar to show the scale of the amplitudes, a cross
    representing the position of the incoming X ray beam in the detector (put at
    (7.5, 7.5) if not calculated). Finallly it offers the possibility to change
    the scale if one has extrapolated the data in more pixels, and to save
    and/or show the output in the console.
    '''
    axes=plt.gca()
    # The matrix representation holds the 'first' pixel in the top left corner,
    # which implies a y axis going down with increasing values. This being
    # contrary to the plot conventions, we have to flip up and down (with
    # numpy's flipud function) the image and y axis tick labels.
    # The fact that we want to add a grid to the plot appears when we specify
    # "edgecolors" and linewidth ("lw") arguments in plt.pcolormesh.
    plt.pcolormesh(np.flipud(img), cmap='hot', edgecolors='gray', lw=0.08)
    axes.set_yticks(np.flipud(np.linspace(0, fig_scale*16, fig_scale*16+1)), minor=True)
    axes.set_aspect('equal') # This constrains the pixels to be square.
    axes.set_xticklabels('')
    axes.set_xticks(np.linspace(0, fig_scale*16, fig_scale*16+1), minor=True)
    axes.set_xticklabels([str(x) for x in np.arange(0, fig_scale*16)], minor=True)
    axes.set_yticklabels('')
    axes.set_yticklabels([str(y) for y in np.arange(0, fig_scale*16)], minor=True)
    plt.colorbar(label='Nombre d\'électrons captés (UA)')
    plt.scatter(fig_scale*center[0], fig_scale*(16-center[1]), s=100, marker='x')
    plt.xlabel('N° des pixels selon l\'horizontale')
    plt.ylabel('N° des pixels selon la verticale')
    if save :
        plt.savefig(name+'.png', dpi=fig_dpi)
    if show :
        plt.show()
    plt.clf()


def visualization(batch, center, save_all, show_all, save_dir='Images/test/'):
    ''' This next function loops over a whole batch folder and plots all the events
    in this batch without sorting anything. It gives the possibility to show and/or
    save the images.
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for (img, file_nb, k) in events_browsing(batch):
        heatmap(img, center, name=save_dir+str(file_nb)+'_'+str(k), save=save_all, show=show_all)


def visualization_seuils(batch, center, save_all, show_all, save_dir='Images/test/'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for (img, file_nb, k) in events_browsing(batch):
        heatmap(img, center, name=save_dir+str(file_nb)+'_'+str(k), save=save_all, show=show_all)


if __name__ == '__main__':
    batch = 'SW2(2017.08.10_11.46.52)'
    center = (7.5, 7.5)
    visualization_seuils(batch, center, False, True)

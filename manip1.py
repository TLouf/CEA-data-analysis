# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:58:07 2017

@author: tlouf

This algorithm is used to analyze the data acquired during a test, by first
converting it to fits, then visualizing it. Finally, if the events look
appropriate, the user can plot an energy spectrum, considering only the events
above a given multiplicity, and with an energy threshold. This allows to see in 
more detail the most interesting part of the spectrum, without having any 
possible saturated events expanding the spectrum in too wide a range.
"""

from hexa_to_fits import hexa_fits
import basic_fits as bf
import numpy as np
import os
import matplotlib.pyplot as plt


#%%
def histogram(batch, edge_ok=True, mult_min=9, threshold=np.inf):
    ''' This function builds an array containing the sum of the amplitudes of all
    the pixels of each event contained in the batch of fits files.
    '''
    Lcounts = []
    Lmult = []

    for (img, file_nb, k) in bf.events_browsing_seuils(batch):
        track_on_edge = (True in (img[0,:] > 0) or True in (img[:,0] > 0) or
                        True in (img[15,:] > 0) or True in (img[:,15] > 0))
        counts = np.sum(img)
        mult = len(img.nonzero()[0])
        if ((edge_ok or not track_on_edge) and len(img.nonzero()[0]) > mult_min
            and counts < threshold):
            Lcounts.append(counts)
            Lmult.append(mult)

    return np.array(Lcounts), np.array(Lmult)


def histo_plot(batch, Lcounts, nb_steps):
    ''' This function actually plots the histogram from the array containing
    the number of counts for each event built by "histogram".
    '''
    Emax = np.max(Lcounts)
    Emin = np.min(Lcounts)
    step = (Emax-Emin)/nb_steps
    Ebounds = np.linspace(Emin, Emax+step, nb_steps+2)
    rect_height = np.zeros(nb_steps+2)


    for count in Lcounts:
        rect_height[int((count-Emin)//step)+1] += 1

    axes = plt.gca()
    axes.step(Ebounds, rect_height)

    if not os.path.exists('Gain/'+batch):
        os.makedirs('Gain/'+batch)

    Umax = Ebounds[np.argmax(rect_height)] - step/2
    plt.savefig('Gain/'+batch+'/Umax='+str(int(Umax*10)/10)+'&step='
                +str(int(step*100)/100)+'.png', dpi=300)
    plt.show()
    return Umax


#%%

if __name__ == "__main__":
    batch = 'SW2(2017.09.13_11.38.49)'
    center = (7.5, 7.5)
    save_dir = 'Images/MM4_75-25/p=300mbar/'

    hexa_fits(batch)
    bf.visualization_seuils(batch, center, False, True, save_dir=save_dir)
    Lcounts, Lmult = histogram(batch, edge_ok=False, mult_min=0, threshold=8000)
    histo_plot(batch, Lcounts, 40)
    histo_plot(batch, Lmult, 40)


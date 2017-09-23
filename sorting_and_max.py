# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 16:36:40 2017

@author: tlouf

This module contains a few functions used to sort out the events resulting
from a photo-electric ionization, or from a Compton scattering, and also a
function used to determine more precisely than with pixels of a 16x16 image the
position of the maximum in a given image.
"""

import numpy as np
from math_functions import gauss2d
from scipy.optimize import curve_fit


def crop(img, xmax, ymax, scope):
    ''' This function returns a cropped version of the input image. It only keeps
    a square with 2*scope+1 pixels sides, centered around the pixel at the
    (xmax, ymax) coordinate. If this position is too close to one or two edges of
    the image, it puts 0 where we don't have any data. We also return
    the distance at which this position is from each edge.
    '''
    rxg = scope
    ryh = scope
    rxd = scope
    ryb = scope

    while xmax-rxg < 0:
        rxg -= 1
    while ymax-ryh < 0:
        ryh -= 1
    while xmax+rxd > 15:
        rxd -= 1
    while ymax+ryb > 15:
        ryb -= 1

    img2 = np.zeros((1+2*scope, 1+2*scope))
    img2[scope-ryh:scope+ryb+1, scope-rxg:scope+rxd+1] = img[ymax-ryh:ymax+ryb+1, xmax-rxg:xmax+rxd+1].copy()
    return img2, (ryh,ryb,rxg,rxd)


def true_max_pos(img, xmax, ymax, method='None', scope=2):
    '''  This function outputs a more precise position of the maximum than the raw
    information we have with the image, which is the position of the pixel where
    we have the maximum. To do this, two methods are provided. From a cropped-
    around-the-center version of the image with a certain scope, we either compute
    the barycenter of this part of the image, or fit it to a 2D gaussian
    distribution.
    '''
    # If the maximum is located near an edge, we won't get better results with
    # the 2 methods below, because we'll insert 0s when cropping the image,
    # and this will "repulse" the maximum from the edge. We thus reduce the
    # scope of the cropping, and if this is not enough we then just return the
    # maximum's position which was input.
    if method == 'None':
        return xmax, ymax

#    if (xmax in np.arange(1,scope+1)) or (ymax in np.arange(1,scope+1)) or (xmax in np.arange(15-scope,15)) or (ymax in np.arange(15-scope,15)):
    if (xmax in np.arange(0,scope)) or (ymax in np.arange(0,scope)) or (xmax in np.arange(16-scope,16)) or (ymax in np.arange(16-scope,16)):
        '''Premiers tests si pixels morts sur le tour, deuxièmes tests sinon'''

        scope=1
#        if (xmax in [1,14]) or (ymax in [1,14]):
        if (xmax in [0,15]) or (ymax in [0,15]):
            method = 'None'
            return xmax,ymax

    if method == 'barycenter':
        numx = 0
        denx = [0]
        numy = 0
        deny = [0]

        for n in range(-scope, scope+1):
            for k in range(-scope, scope+1):
                denx[n+scope] += img[ymax+k, xmax+n]
                deny[n+scope] += img[ymax+n, xmax+k]
            numx += (xmax+n)*denx[n+scope]
            numy += (ymax+n)*deny[n+scope]
            denx.append(0)
            deny.append(0)
        xmax = numx/sum(denx)
        ymax = numy/sum(deny)

    if method == 'gaussfit':
        x0, y0 = xmax, ymax
        img2, r = crop(img, x0, y0, scope)
        zobs = img2.ravel()
        x, y = np.mgrid[0:2*scope+1, 0:2*scope+1]
        yx = np.vstack((y.ravel(), x.ravel()))
        guess = [10000, scope, scope, 1, 1, 0]
        constraints = (np.array([0,0,0,0,0,-np.inf]), np.array([np.inf,2*scope,2*scope,4,4,np.inf]))

        # We very rarely get a RunTimeError, in which case we'll just pass and
        # return the input maximum position.
        try:
            pred_params, uncert_cov = curve_fit(gauss2d, yx, zobs, p0=guess, bounds=constraints)
            # The maximum's position is deduced from the gaussian distribution's center
            ymax, xmax = y0+pred_params[2]-(2*scope-r[0]), x0+pred_params[1]-(2*scope-r[2])
        except RuntimeError:
            pass

    return xmax, ymax


def one_event(img, xmax, ymax, r):
    ''' This function determines whether the image we are looking at is the
    result of a multiple events and not only one, in the rare case a secondary
    electron ionized another gas molecule.
    '''
    img2 = img.copy()
    (ryh,ryb,rxg,rxd) = crop(img2, xmax, ymax, r)[1]

    for x in range(xmax-rxg, xmax+rxd+1):
        for y in range(ymax-ryh, ymax+ryb+1):
            img2[y,x] = 0

    # max2 is the maximum when we've removed a square of 2*r+1 pixels sides
    # around the maximum
    max2 = np.max(img2)
    posmax2 = np.argmax(img2)
    xmax2 = posmax2%16
    ymax2 = posmax2//16
    xmax3 = []
    ymax3 = []
    max3 = 0

    if ymax-ryh-1 >= 0:
        for x in range(xmax-rxg, xmax+rxd+1):
            xmax3.append(x)
            ymax3.append(ymax-ryh-1)

    if ymax+ryb+1 <= 15:
        for x in range(xmax-rxg, xmax+rxd+1):
            xmax3.append(x)
            ymax3.append(ymax+ryb+1)

    if xmax-rxg-1 >= 0:
        for y in range(ymax-ryh, ymax+ryb+1):
            xmax3.append(xmax-rxg-1)
            ymax3.append(y)

    if xmax+rxd+1 <= 15:
        for y in range(ymax-ryh, ymax+ryb+1):
            xmax3.append(xmax+rxd+1)
            ymax3.append(y)

    ymax3 = np.array(ymax3)
    xmax3 = np.array(xmax3)
    k = np.argmin((ymax3-ymax2)**2+(xmax3-xmax2)**2)
    # max3 is the closest pixel to max2 which is on the edge of the removed square
    max3 = img[ymax3[k],xmax3[k]]
    # To consider that there is indeed only one event in the image, max2 has to
    # be sufficiently low compared to max3 and sufficiently close to max
    return max3/max2 > 0.9 and (xmax2-xmax)**2+(ymax2-ymax)**2 <= 2*(r+3)**2


def photoelec_sort(img, xmax, ymax, center):
    ''' Returns a boolean which indicates whether the input image
    is the result of a photoelectric interaction and if not, the reason why not
    (this output was used to put each condition to the test and refine the
    constraints).
    '''
    photoelec = False

    ring = ((ymax-center[1])**2+(xmax-center[0])**2 >= 25)
    # We also want the track to start near this center, because otherwise
    # we are not observing the result of a photoelectric interaction with the
    # X ray beam, but a "secondary" ionization
    centered = len(img[6:9, 6:9].nonzero()[0]) >= 4
    reason = 'not in ring or not centered'

    # It happened once that an image had negative values, this seems to be
    # extremely rare but we check this doesn't happen by ensuring that the
    # minimum is 0. We also check that the track has a sufficient multiplicity.
    # These very quick tests are also the ones which eliminate the most events,
    # and this is why there are done first.
    if ring and centered and np.min(img) >= 0 and len(img.nonzero()[0]) > 30:
        xedge=[]
        yedge=[]

        for x in range (16):
            for y in range (16):

                if img[y,x] != 0:
                    img3 = crop(img, x, y, 1)[0]

                    if img3[0,1]==0 or img3[1,0]==0 or img3[2,1]==0 or img3[1,2]==0:
                        xedge.append(x)
                        yedge.append(y)

        xedge = np.array(xedge)
        yedge = np.array(yedge)

        # The piece of code above finds the pixels of the track which are on
        # its edge, we are then going to compute a variance of the distance
        # from the position of the maximum to each of these edges, which
        # caracterizes the "circularity" of the track. Some testing helped
        # determine that a minimum of 30 for this variance gave the best sorting
        true_xmax, true_ymax = true_max_pos(img, xmax, ymax, method='barycenter')
        dist_to_max = (xedge-true_xmax)**2 + (yedge-true_ymax)**2
        variance = 1/len(xedge)*np.sum((np.mean(dist_to_max)-dist_to_max)**2)
        circular = variance >= 30
        reason = 'circular_var='+str(int(variance*10) / 10)
        photoelec = circular

        if circular :
            # Finally, when all the remaining conditions have been tested, we
            # check that there is only one physical event recorded, which is
            # the rarest occurrence of the ones tested.
            reason = 'more than 1 event'
            one_event_test = one_event(img, xmax, ymax, 3) and one_event(img, xmax, ymax, 2) and one_event(img, xmax, ymax, 1)
            photoelec = one_event_test

    return photoelec, reason


if __name__ == "__main__":
    pass
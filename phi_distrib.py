# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:46:02 2017

@author: tlouf
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import time
import basic_fits as bf
import math_functions as mth_fct
import sorting_and_max as srt_max
import csv_manips as csvm
import os

#%%

def phidet(x_sample, y_sample, xmax, ymax, weights, deg=3, polyshow=False):
    ''' For a given event, returns the azimuthal angle of ejection of the photo
    electron
    '''
    # We first perform a weighted 2D polynomial fit over the image, in order
    # to, in the end, get the coefficients (c) of a polynome whose graph fits
    # the trajectory of our photo electron. We use a linear least squares
    # method to determine thhese coefficients with the appropriate matrices.
    XX = np.column_stack([x_sample**(deg-i) for i in range(deg)]).T
    Vweigths = np.sqrt(weights)
    Vweights2 = np.tile(Vweigths, (deg, 1))
    Xw = np.multiply(XX, Vweights2)
    yw = y_sample * Vweigths
    c = np.flipud(np.append(np.linalg.lstsq(Xw.T, yw)[0], 0))

    # We compute the list of the polynome's derivative's coefficients
    c_prime = np.array([c[i]*i for i in range(1,deg+1)])

    phi_sample = np.angle(x_sample+1j*y_sample)
    phi_sample_moy = (np.angle(np.mean(np.cos(phi_sample))+1j*np.mean(np.sin(phi_sample)), deg=True)+360)%360
    # We have to determine the direction of the gradient
    phi_grad = (np.angle((1+1j*c[1], -1+1j*c[1], 1-1j*c[1], -1-1j*c[1]), deg=True)+360)%360
    i = np.argmin((np.abs(phi_sample_moy-phi_grad), np.abs(phi_sample_moy-phi_grad-360)))%4
    # We return 2 angles, each being the average of the result of 3 methods,
    # this allows to "smooth" few exceptional cases which are treated poorly
    # by the direct calculation of the gradient at the origin
    phi = phi_grad[i]
    phi2 = phi_grad[i]

    y_grad = mth_fct.poly1d(-1, c_prime, deg-1)
    phi_grad = (np.angle((1+1j*y_grad, -1+1j*y_grad, 1-1j*y_grad, -1-1j*y_grad), deg=True)+360)%360
    i = np.argmin((np.abs(phi_sample_moy-phi_grad), np.abs(phi_sample_moy-phi_grad-360)))%4
    phi += phi_grad[i]

    y_grad = mth_fct.poly1d(1, c_prime, deg-1)
    phi_grad = (np.angle((1+1j*y_grad, -1+1j*y_grad, 1-1j*y_grad, -1-1j*y_grad), deg=True)+360)%360
    i = np.argmin((np.abs(phi_sample_moy-phi_grad), np.abs(phi_sample_moy-phi_grad-360)))%4
    phi += phi_grad[i]

    x_grad = np.sign(np.cos(phi_sample_moy*np.pi/180))
    phi2 += (np.angle(x_grad + 1j*mth_fct.poly1d(x_grad, c, deg), deg=True)+360)%360
    phi1, phi2 = phi/3, phi2/2

    # There's the option to show the polynomial fit and the angle which
    # resulted from the process carried out above
    if polyshow:
        x_plot = np.linspace(-centers[cas][1],15-centers[cas][1],100)
        y_plot = mth_fct.poly1d(x_plot, c, deg)
        plt.plot(x_plot, y_plot)
        plt.scatter(x_sample, y_sample, c=weights, cmap='hot', s=150, vmin=0)
        ax = plt.gca()
        ax.set_facecolor('k')
        ax.set_ylim([-8,8])
        ax.set_xlim([-8,8])
        ax.set_aspect('equal')
        plt.grid()
        plt.xlabel('x')
        plt.ylabel('y')
        ax.set_xticks(ax.get_yticks())
        if not os.path.exists('Polyfit/'+folders[cas]+'/'):
            os.makedirs('Polyfit/'+folders[cas]+'/')
        plt.savefig('Polyfit/'+folders[cas]+'/'+str(int((time.time()-start_time)*10)/10)
                    +'&phi='+str((int(phi1*10)/10, int(phi2*10)/10)), dpi=300)
        plt.show()

        plt.clf()


    return phi1, phi2, c

#%%

def histogram(batch, center, save_all, show_all, max_det, fit_deg=3):
    '''Variables declared as global so that we are able to stop the execution
    and still save the lists. Allows to do preliminary tests of the algorithm
    without running it on all the data.
    '''
    global Lposmax
    Lposmax = []
    global Lphi1
    Lphi1 = []
    global Lphi11
    Lphi11 = []
    global Lphi2
    Lphi2 = []
    global Lphi3
    Lphi3 = []

    photoelec_dir = 'Photoelec/'+batch+'/'
    if (not os.path.exists(photoelec_dir)) and save_all:
        os.makedirs(photoelec_dir)

    for (img, file_nb, k) in bf.events_browsing(batch):
        # 'img' is 16x16 pixels, and np.argmax returns the position of the max
        # as though img was a list, so posmax takes values between 0 and 255
        posmax = np.argmax(img)
        xmax = posmax%16
        ymax = int(posmax/16)
        photoelec, reason = srt_max.photoelec_sort(img, xmax, ymax, center)

        if photoelec :
            true_xmax, true_ymax = srt_max.true_max_pos(img, xmax, ymax, method=max_det)

            if save_all or show_all:
                file_name = str(file_nb)+'_'+str(k)+'_max='+str((int(true_ymax*100)/100, int(true_xmax*100)/100))
                bf.heatmap(img, center, name=photoelec_dir+file_name, save=save_all, show=show_all)

            Ly, Lx = img.nonzero()
            Lpix = [pair for pair in zip(Lx, Ly)]
            weights = []
            for (x_pix, y_pix) in Lpix:
                 weights.append(img[y_pix, x_pix])
            weights = np.array(weights)

            Ly, Lx = centers[cas][1]-Ly, Lx-centers[cas][0]
            lenx = max(Lx)-min(Lx)
            leny = max(Ly)-min(Ly)
            xmax, ymax = xmax-centers[cas][0], centers[cas][1]-ymax
            distances = (Lx)**2+(Ly)**2
            phi_s = np.angle(Lx+1j*Ly)
            x_start_avg = np.average(np.cos(phi_s), weights=weights)
            y_start_avg = np.average(np.sin(phi_s), weights=weights)
            phi_s_moy = (np.angle(x_start_avg+1j*y_start_avg, deg=True)+360)%360
            weights2 = weights/(distances**(1/3)+1)

            # We determined 'phi_s_moy', which is the weighted average angle
            # of the pixels which recorded some signal, in order to determine
            # whether our track had more of a vertical or horizontal direction.
            # Thus, we switch the y and x axes of images whose track is vertical,
            # and we can fit it better to a polynome, which, because it is a
            # function, can't have more than 1 image (y) for 1 inverse image (x).
            if (phi_s_moy > 240 and phi_s_moy < 300 or phi_s_moy > 60 and phi_s_moy < 120 or
                (phi_s_moy > 210 and phi_s_moy < 330 or phi_s_moy > 30 and phi_s_moy < 150) and leny > lenx):
                phi1, phi11, c = phidet(Ly, Lx, ymax, xmax, weights2, deg=fit_deg, polyshow=show_all)
                phi1 = (360+90-phi1) % 360
                phi11 = (360+90-phi11) % 360
                Lphi1.append(phi1)
                Lphi11.append(phi11)
            else:
                phi1, phi11, c = phidet(Lx, Ly, xmax, ymax, weights2, deg=fit_deg, polyshow=show_all)
                Lphi1.append(phi1)
                Lphi11.append(phi11)


#%%

def phi_distrib(phi, A, B, phi0):
    '''Returns the theoretical probability density that a linearly polarized
    beam created a photo electron ejected with a given phi angle.
    '''
    return A+B*(np.cos((phi-phi0)*np.pi/180))**2


def histo_fit(kind, number, step):
    '''Fits the phi distribution defined above to the histogram built from
    experimental data, and plots both the fit and the histogram.
    '''
    Lphi = csvm.read_csv(folders[cas]+'_'+kind+'_'+number)
    nb_steps = 360//step
    phi_bounds = np.linspace(0, 360, nb_steps+1)
    rect_height = np.zeros(nb_steps)
    for phi in Lphi:
         rect_height[int(phi//step)] += 1
    axes = plt.gca()
    axes.step(phi_bounds, np.insert(rect_height, 0, rect_height[-1]))
    phi_bounds = phi_bounds[:-1]+step/2
    if step >= 10:
        axes.errorbar(phi_bounds, rect_height, np.sqrt(rect_height), fmt='none', ecolor='k', capsize=3)
    axes.set_xticks(np.linspace(0, 360, 9))
    guess = [500, 3000, 45]
    [A, B, phi0], uncert_cov = opt.curve_fit(phi_distrib, phi_bounds, rect_height, p0=guess)
    mu = B/(B+2*A)
    phi_plot = np.linspace(0, 360, 1000)
    plt.plot(phi_plot, phi_distrib(phi_plot, A, B, phi0), 'r')
    rect_height = np.array(rect_height)
    r2 = 1-np.sum((rect_height-phi_distrib(phi_bounds, A, B, phi0))**2)/np.sum((rect_height-np.mean(rect_height))**2)
    name = '(A,B,phi0,mu)='+str((int(A*10)/10, int(B), int(phi0*10)/10, int(mu*100)/100))+'&r2='+str(int(r2*100)/100)+'&step='+str(step)
    histo_dir = 'Histograms/'+folders[cas]+'/'
    if not os.path.exists(histo_dir):
        os.makedirs(histo_dir)
    plt.xlabel('Angle d\'éjection (en °)')
    plt.ylabel('Nombre de traces')
    plt.savefig(histo_dir+'Histofit_'+kind+'_'+number+'_'+name+'deg.pdf')#, dpi=300)
    plt.show()
    return A, B, phi0, mu

#%%

if __name__ == "__main__":
    start_time = time.time()
    cas = 1
    centers = np.array(((7.4, 7.4), (7.45, 7.35), (7.5, 7.3), (8.1, 7.4))) #(x,y)
    folders = ('8 keV_0 degrés', '8 keV_45 degrés2', '10 keV_0 degrés', '10 keV_45 degrés')
    save_dir = 'Images/'+folders[cas]+'/'

#    bf.visualization(folders[cas], centers[cas], True, True, save_dir=save_dir)
#    histogram(folders[cas], centers[cas], False, False, 'None', fit_deg=2)

    nb = '36'
    #csvm.write_csv(Lphi1, folders[cas]+'_Lphi1_'+nb)
    #csvm.write_csv(Lphi11, folders[cas]+'_Lphi11_'+nb)


    step = 30

    A, B, phi0, mu = histo_fit('Lphi1', nb, step)
    A, B, phi0, mu = histo_fit('Lphi11', nb, step)
    #print(A, B, phi0, mu)


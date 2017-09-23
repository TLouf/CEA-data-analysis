# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:58:07 2017

@author: tlouf

This module converts the data acquired by IdefX in a telemetric (.tm)
format, which is binary, to the .fits format, widely used for applications in
astronomy. This format is then easily readable thanks to the astropy.fits
library (see how this is done is the "basic_fits" module).
"""

import os
import glob
import numpy as np
from astropy.table import Table


def hexa_fits(batch):
    '''The .tm files contain binary data. They can be divided in blocks of 4 bytes,
    the first byte of each block containing the data we seek (the frame and pixel
    number and its associated amplitude), and the 3 others all being set to 0, with
    the exception of the second one whose value is 1 when we switch from one frame
    to another. The first byte of the third block of a frame should have a value of
    128, else there may have been an issue in the acquisition.
    '''
    # The batch of tm files you wish to convert should be placed in its own
    # folder in the TM folder
    tm_files = glob.glob('TM/'+batch+'/*.tm')
    for k, file in enumerate(tm_files):
        print(file)

        with open(file, 'rb') as f:  # 'rb' instead of 'r' to read binary data
            data = f.read()

        # Lend_i is the list of indices of the end of each frame. There is no
        # need to add len(data) at the end because the first and last frames
        # may not be complete.
        Lend_i = [i*4 for i in range(len(data)//4-1) if data[i*4+1] == 1]
        nFin = len(Lend_i)
        Lgood_frames_i = [n for n in range(nFin) if data[Lend_i[n]+12] == 128]
        # The "real' start of the data for a specific frame is stored right
        # after the block containing the byte at 128
        Lstart_i = [Lend_i[frame]+12 for frame in Lgood_frames_i]
        eventN = len(Lstart_i)
        structure_tot = np.zeros((5, eventN*256))
        num_event = 0

        for j in range(eventN-1):
            # Right after each index stored in Lstart_i, we find the data
            # we are after with leaps of a multiple of 4 bytes. The formulas to
            # extract the values are of course specific to the Caliste.
            frameN = data[Lstart_i[j]+8]*256 + data[Lstart_i[j]+12]
            time = (167772.16*data[Lstart_i[j]+16] + 655.36*data[Lstart_i[j]+20]
                    + 2.56*data[Lstart_i[j]+24] + 0.01*data[Lstart_i[j]+28])
            multiplicity = data[Lstart_i[j]+36]
            pos_pix = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,
                       1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31]

            for l in range(multiplicity):
                position = data[Lstart_i[j]+40+12*l]
                asicN = np.right_shift(position, 5)
                channelN = position % 32
                p = np.where(np.array(pos_pix) == channelN)[0][0]
                pixN = p + (7-asicN)*32
                amp = data[Lstart_i[j]+44+12*l]*256 + data[Lstart_i[j]+48+12*l]

                structure_tot[0][num_event] = int(frameN)
                structure_tot[1][num_event] = time
                structure_tot[2][num_event] = int(pixN)
                structure_tot[3][num_event] = int(multiplicity)
                structure_tot[4][num_event] = int(amp)
                num_event += 1

        structure = [event for event in structure_tot.T if event[3] != 0]
        # names' order must match structure_tot's lines number
        names = ('NFRAME', 'TIME_us', 'PIXEL', 'MULTIPLICITY', 'AMPLITUDE')
        # We then create a Table object, which is what we intuitively would
        # think it is : it stores the lists in structure in what would be like
        # lines of a table, with a name attributed to each line.
        t = Table(rows=structure, names=names)
        if not os.path.exists('FITS/'+batch):
            os.makedirs('FITS/'+batch)
        # This Table object is convenient because it can be easily converted to
        # a fits file as is done below.
        t.write('FITS/'+batch+'/'+str(k)+'.fits', format='fits')


if __name__ == "__main__":
    batch = 'SW2(2017.07.20_11.00.31)'
    hexa_fits(batch)


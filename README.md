# CEA-data-analysis

This repository contains part of my work at the CEA, during which I analyzed data from a gaseous detector called *Micromegas*, a detector still in development which aims at being an X-ray spectro-polarimeter for astrophysics (see this [article](http://iopscience.iop.org/article/10.1088/1748-0221/11/04/P04016/pdf) co authored by my internship tutor). In (very) short, this detector records events on a pixellated electronics, and my job was to analyze them. In the following, I describe (extremely briefly) the different steps of this analysis.

## Converting the telemetric files to fits files

The output of the acquisition is a batch of telemetric (.tm) files, binary data formatted in a specific way which needs to be compressed and made more readable. To achieve this, we convert them to .fits files, a format which is widely used in astrophysics to store the data coming from detectors. This operation can be done thanks to the *hexa_to_fits.py* module : after putting the directory containing the batch of .tm files you want to convert into the TM/ directory, just copy the name of the batch into the 'batch' variable's definition.

## Constructing images of the events

The *basic_fits.py* module provides convenience functions to construct images and visualizing them, and also a generator that lets you iterate through all the events of a batch.

## Analyzing the data

Finally, not to enter into too much detail, *sorting_and_max.py*, *phi_distrib.py* and *manip1.py* are used to extract meaningful information from the data, such as the gain of the detector or its modulation factor.


I uploaded a bit of data just so you can make some tests, including a few tm files, fits files of an interesting batch for *phi_distrib*, csv files containing results of the analysis performed by *phi_distrib*, and histograms of both energy spectra and phi distributions fit to the theory using a least squares regression.

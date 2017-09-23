# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 16:44:31 2017

@author: tlouf

This module defines some mathematical functions which are used in other
modules mainly to fit data to a distribution.
"""

import numpy as np


def gauss2d(xy, amp, x0, y0, sigma_x, sigma_y, theta):
    (x, y) = xy
    a = np.cos(theta)**2 / (2*sigma_x**2) + np.sin(theta)**2 / (2*sigma_y**2)
    b = -np.sin(2*theta) / (4*sigma_x**2) + np.sin(2*theta) / (4*sigma_y**2)
    c = np.sin(theta)**2 / (2*sigma_x**2) + np.cos(theta)**2 / (2*sigma_y**2)
    return (amp * np.exp(-(a*(x-x0)**2 - 2*b*(y-x0)*(y-y0) + c*(y-y0)**2))).ravel()


def poly1d(x, c, deg):
    x = [x**i for i in range(deg+1)]
    return np.dot(c.T, x)


if __name__ == "__main__":
    pass

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 16:02:44 2017

@author: tlouf

This module allows some simple manipulation of csv files, used to store data
originally in the form of a list
"""

import csv


def write_csv(data, file):
    '''From data, which is a list, this function will create a csv file with a
    single row containing this data
    '''
    with open('CSV/'+file+'.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)

def read_csv(file):
    ''' Allows to read csv files in the format of the ones written with write_csv
    and  memorize them in the form of a list
    '''
    with open('CSV/'+file+'.csv', newline='') as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        return [row for row in reader][0]


if __name__ == "__main__":
    a=read_csv('seuils')


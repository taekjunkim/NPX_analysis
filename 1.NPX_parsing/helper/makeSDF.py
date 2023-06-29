#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 14:20:50 2019

sdf = makeSDF.getSDF(spkTrain,FS,AllorMean=0);
make spike density function from spike train (columns = time points, rows = different trials), 
                                 FS (sample frequency: e.g., 1000),
                                 AllorMean: 0 for mean SDF, 1 for each trial SDF

@author: taekjunkim
"""
import numpy as np;

def getSDF(spkTrain,FS,AllorMean=0):
    # Make gaussian kernel window
    sigma = 5;
    t = np.arange(-3*sigma,3*sigma+1);

    y = (1/sigma*np.sqrt(np.pi*2)) * np.exp(-(t**2)/(2*sigma**2));
    window = y[:];
    window = window/np.sum(window);

    # convolution
    sdf = np.zeros(np.shape(spkTrain));
    for i in np.arange(np.shape(spkTrain)[0]):
        convspike = np.convolve(spkTrain[i,:],window);
        pStart = int(np.floor(len(window)/2));
        pEnd = int(np.floor(len(window)/2)+np.shape(spkTrain)[1]);
        convspike = convspike[pStart:pEnd];
        sdf[i,:] = convspike;
    sdf = sdf*FS;

    if AllorMean==0:
        sdf = np.mean(sdf,axis=0);
    
    return sdf;        
        
        

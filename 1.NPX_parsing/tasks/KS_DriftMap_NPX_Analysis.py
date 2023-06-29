#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:38:14 2019

KS_DriftMap_NPX_Analysis.py

1. Get spikeTimes, spikeAmps, spikeDepths, spikeSites for every spike

@author: taekjunkim
"""
#%%
import sys
import matplotlib.pyplot as plt
import pandas as pd; 
import numpy as np; 
import json;
import gzip;


#%%
def ismember(a_vec, b_vec):
    """ MATLAB equivalent ismember function """

    bool_ind = np.isin(a_vec,b_vec)
    common = a_vec[bool_ind]
    common_unique, common_inv  = np.unique(common, return_inverse=True)     # common = common_unique[common_inv]
    b_unique, b_ind = np.unique(b_vec, return_index=True)  # b_unique = b_vec[b_ind]
    common_ind = b_ind[np.isin(b_unique, common_unique, assume_unique=True)]
    return bool_ind, common_ind[common_inv]

def loadKSdir(imec_dataFolder):
    sampRate = 30000.0; 
    spikeStruct = dict(); 
    spikeTimes = np.load(imec_dataFolder+'spike_times.npy')/sampRate; 
    spikeTemplates = np.load(imec_dataFolder+'spike_templates.npy'); 
    spikeTempAmps = np.load(imec_dataFolder+'amplitudes.npy'); 
    pcFeat = np.load(imec_dataFolder+'pc_features.npy');     
    pcFeatInd = np.load(imec_dataFolder+'pc_feature_ind.npy');       
    spikeStruct['pcFeatInd'] = pcFeatInd;   

    clu = np.load(imec_dataFolder+'spike_clusters.npy'); 
    cids = np.unique(clu); 
    cgs_df = pd.read_csv(imec_dataFolder+'cluster_group.tsv', sep='\t');
    noise_cluster = cgs_df['cluster_id'][cgs_df['group']=='noise'].values; 

    spikeStruct['spikeTimes'] = spikeTimes[~ismember(clu,noise_cluster)[0]]; 
    spikeStruct['spikeTemplates'] = spikeTemplates[~ismember(clu,noise_cluster)[0]]; 
    spikeStruct['spikeTempAmps'] = spikeTempAmps[~ismember(clu,noise_cluster)[0]]; 
    spikeStruct['pcFeat'] = pcFeat[~ismember(clu,noise_cluster)[0],:,:]; 
    #spikeStruct['pcFeatInd'] = pcFeatInd[~ismember(cids,noise_cluster)[0],:]; 

    spikeStruct['clu'] = clu[~ismember(clu,noise_cluster)[0]]; 
    spikeStruct['cgs'] = cgs_df['cluster_id'][cgs_df['group']!='noise'].values; 
    spikeStruct['cids'] = cids[~ismember(cids,noise_cluster)[0]]; 

    coords = np.load(imec_dataFolder+'channel_positions.npy'); 
    spikeStruct['xcoords'] = coords[:,0]; 
    spikeStruct['ycoords'] = coords[:,1]; 

    spikeStruct['temps'] = np.load(imec_dataFolder+'templates.npy'); 
    spikeStruct['winv'] = np.load(imec_dataFolder+'whitening_mat_inv.npy');    

    return spikeStruct; 

#%%
def main(app):

    imec_filename = app.imec_file; 
    imec_dataFolder = imec_filename[:(imec_filename.rfind('/')+1)]; 

    ### Load KS dir
    sp = loadKSdir(imec_dataFolder); 
    print('Spike structure has been loaded'); 

    ycoords = sp['ycoords']; 
    pcFeat = sp['pcFeat']; 
    pcFeat = np.squeeze(pcFeat[:,0,:]);    # take first PC only
    pcFeat[pcFeat<0] = 0;    # some entries are negative, but we don't really want to push the CoM away from there.
    pcFeatInd = sp['pcFeatInd']; 
    spikeTemps = sp['spikeTemplates']; 

    temps = sp['temps']; 
    winv = sp['winv']; 
    tempScalingAmps = sp['spikeTempAmps']; 
    spikeTimes = sp['spikeTimes']; 

    ### Compute center of mass of these features
    spikeFeatInd = pcFeatInd[spikeTemps,:];     
    spikeFeatYcoords = np.squeeze(ycoords[spikeFeatInd]);  # 2D matrix of size #spikes x 12 
    spikeDepths = np.sum(np.multiply(spikeFeatYcoords,pcFeat**2),axis=1)/np.sum(pcFeat**2,axis=1);  

    """
    for plotting, we need the amplitude of each spike, both so we can draw a
    threshold and exclude the low-amp noisy ones, and so we can color the
    points by the amplitude
    """
    # tempsUnW and spikeAmps from templatePositionsAmplitudes.m 
    # unwhiten all the templates
    tempsUnW = np.zeros(np.shape(temps)); 
    for t in range(np.shape(temps)[0]):
        tempsUnW[t,:,:] = np.matmul(np.squeeze(temps[t,:,:]),winv); 

    # The amplitude on each channel is the positive peak minus the negative
    tempChanAmps = np.squeeze(np.max(tempsUnW,axis=1))-np.squeeze(np.min(tempsUnW,axis=1)); 

    # The template amplitude is the amplitude of its largest channel (but see below for true tempAmps)
    tempAmpsUnscaled = np.max(tempChanAmps,axis=1);

    # assign all spikes the amplitude of their template multiplied by their 
    # scaling amplitudes (templates are zero-indexed)
    spikeAmps = tempAmpsUnscaled[spikeTemps]*tempScalingAmps; 

    print('Amplitudes, Depths information has been computed'); 

    #max_site = np.argmax(np.max(np.abs(tempsUnW),axis=1),axis=1); 
    #spikeSites = max_site[spikeTemps]; 

    ### Plot driftmap
    nColorBins = 20; 
    ampRange = np.quantile(spikeAmps, [0.1, 0.9]);    
    colorBins = np.linspace(ampRange[0], ampRange[1], nColorBins);    
    gray_cmap = plt.cm.get_cmap('gray',nColorBins); 
    gray_colors = [gray_cmap(19-i) for i in range(20)]; 

    plt.figure(figsize=[8,6]); 
    for i in range(nColorBins-1):
        theseSpikes = np.where((spikeAmps>=colorBins[i]) & (spikeAmps<=colorBins[i+1]))[0]; 
        plt.plot(spikeTimes[theseSpikes], spikeDepths[theseSpikes], '.', color=gray_colors[i]); 
    plt.xlabel('Time in seconds'); 
    plt.ylabel('Distance from the probe tip'); 
    plt.show(); 

class NumpyEncoder(json.JSONEncoder):
    # Special json encoder for numpy types 
    """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)z`
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj,np.ndarray): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


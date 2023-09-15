#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:38:14 2019

ClutterStim_NPX_Analysis.py

@author: taekjunkim
"""

import matplotlib.pyplot as plt;
import scipy.optimize as opt;
from scipy.ndimage import gaussian_filter1d, gaussian_laplace
from scipy.signal import butter, sosfilt

#import sys;
#sys.path.append('./helper'); 

import numpy as np;
import glob; 

import os; 
from helper import parseTJexperiment_NPX as parse_NPX;
import json;
import gzip;


def main(app):

    #%% This needs to be defined
    dat_filename = app.dat_file; 
    imec_filename = app.imec_file; 
    task_folder = dat_filename[:(dat_filename.rfind('/')+1)]; 
    bin_filename = glob.glob(task_folder+'*.nidq.bin')[0]; 

    task_index_in_combine = int(app.tasknum_lineEdit.text()); 

    #%% get pdOnTS
    markervals, markervals_str = parse_NPX.get_markervals(dat_filename);     
    markerts, pdOnTS, pdOffTS = parse_NPX.get_event_ts(bin_filename, markervals_str); 

    #%% get LFP data from imec_file
    meta_filename = imec_filename[:-3]+'meta';     
    metaDict = parse_NPX.get_metaDict(meta_filename); 
    rawData = parse_NPX.access_rawData(imec_filename, metaDict); 

    ### detect syncOn in the first 20 seconds 
    imSampRate = int(metaDict['imSampRate']); 
    syncCh = 384; 
    syncON = np.where(rawData[syncCh,:imSampRate*20]
                      >np.max(rawData[syncCh,:imSampRate*20])*0.5)[0][0];    

    ### LFP cut, align at pdON
    LFP_TS = (np.arange(np.shape(rawData)[1]) - syncON)/imSampRate; 
    for i in range(len(pdOnTS)):
        tsNow = np.where((LFP_TS>=pdOnTS[i]-0.1) & (LFP_TS<=pdOnTS[i]+0.4))[0]; 
        tsNow = tsNow[:int(0.5*imSampRate)];
        base_level = np.mean(rawData[:384, tsNow[:int(0.1*imSampRate)]], axis=1).reshape(384,1); 
        if i==0:
            LFP_mtx = rawData[:384,tsNow] - np.matmul(base_level,np.ones((1,int(0.5*imSampRate)))); 
        else:
            LFP_mtx += rawData[:384,tsNow] - np.matmul(base_level,np.ones((1,int(0.5*imSampRate)))); 
    LFP_mtx = LFP_mtx/len(pdOnTS); 

    ### Fix ch 191
    LFP_mtx[191,:] = (LFP_mtx[190,:]+LFP_mtx[192,:])/2; 

    ### subtract baseline
    for i in range(np.shape(LFP_mtx)[0]):
        LFP_mtx[i,:] = LFP_mtx[i,:] - np.mean(LFP_mtx[i,:int(0.1*imSampRate)]);     

    ### apply butter bandpass filter
    LFP_mtx = butter_bandpass_filter(LFP_mtx, lowcut=0.3, highcut=250, fs=imSampRate, order=10); 

    ### get_NPX_chpos 
    NPX_chpos = get_NPX_chpos('3B1_staggered'); 

    '''
    ### divide LFP_mtx into 4 groups based on xcoords
    LFP_mtxA = LFP_mtx[np.where(NPX_chpos[:,0]==43)[0],:];  
    LFP_mtxB = LFP_mtx[np.where(NPX_chpos[:,0]==11)[0],:];  
    LFP_mtxC = LFP_mtx[np.where(NPX_chpos[:,0]==59)[0],:];  
    LFP_mtxD = LFP_mtx[np.where(NPX_chpos[:,0]==27)[0],:];              
    '''

    ### divide LFP_mtx into 2 groups based on ycoords
    LFP_mtxA = (LFP_mtx[np.where(NPX_chpos[:,0]==43)[0],:] +   
                LFP_mtx[np.where(NPX_chpos[:,0]==11)[0],:])/2;  
    LFP_mtxB = (LFP_mtx[np.where(NPX_chpos[:,0]==59)[0],:] +   
                LFP_mtx[np.where(NPX_chpos[:,0]==27)[0],:])/2;  
    LFP_mtxC = (LFP_mtxA + LFP_mtxB)/2; 

    ### denoise LFP_mtx with gaussian_filter
    LFP_mtxA = gaussian_filter1d(LFP_mtxA, sigma=5, axis=0, mode='reflect'); 
    LFP_mtxB = gaussian_filter1d(LFP_mtxB, sigma=5, axis=0, mode='reflect'); 
    LFP_mtxC = gaussian_filter1d(LFP_mtxC, sigma=5, axis=0, mode='reflect'); 

    #LFP_mtxA = gaussian_filter1d(LFP_mtxA, sigma=2, axis=1, mode='reflect'); 
    #LFP_mtxB = gaussian_filter1d(LFP_mtxB, sigma=2, axis=1, mode='reflect'); 
    #LFP_mtxC = gaussian_filter1d(LFP_mtxC, sigma=2, axis=1, mode='reflect'); 


    ### compute CSD. negative of the 2nd spatial derivative
    spacing_mm = 0.02; 
    LFP_mtxA2 = -np.diff(LFP_mtxA, n=2, axis=0)/spacing_mm**2; 
    LFP_mtxB2 = -np.diff(LFP_mtxB, n=2, axis=0)/spacing_mm**2; 
    LFP_mtxC2 = -np.diff(LFP_mtxC, n=2, axis=0)/spacing_mm**2;     

    level_max = np.max([np.abs(LFP_mtxA), np.abs(LFP_mtxB), np.abs(LFP_mtxC)]); 
    level_max2 = np.max([np.abs(LFP_mtxA2), np.abs(LFP_mtxB2), np.abs(LFP_mtxC2)])*0.6; 

    ### draw figures; 
    plt.figure(figsize=(12,7)); 

    plt.subplot(2,3,1); 
    plt.imshow(LFP_mtxA, aspect='auto', origin='lower', cmap='jet', vmin = -level_max, vmax = level_max); 
    for i in np.arange(0,96,5):
        plt.plot(LFP_mtxA[i,:]*8/level_max + i,'k'); 
    plt.xticks(np.arange(0,1500,250), labels=np.arange(-100,500,100)); 
    plt.yticks(np.arange(-1,96,5), labels=np.arange(20,3860,200));     
    plt.ylim([0, 95]); 
    plt.title('odd rows. Raw LFP'); 
    plt.xlabel('Time from stimulus onset (ms)')
    plt.ylabel('Distance from NPX tip (micrometer)')

    plt.subplot(2,3,2); 
    plt.imshow(LFP_mtxB, aspect='auto', origin='lower', cmap='jet', vmin = -level_max, vmax = level_max); 
    for i in np.arange(0,96,5):
        plt.plot(LFP_mtxB[i,:]*8/level_max + i,'k'); 
    plt.xticks(np.arange(0,1500,250), labels=np.arange(-100,500,100)); 
    plt.yticks(np.arange(-1,96,5), labels=np.arange(20,3860,200));     
    plt.ylim([0, 95]); 
    plt.title('even rows. Raw LFP'); 
    plt.xlabel('Time from stimulus onset (ms)')
    plt.ylabel('Distance from NPX tip (micrometer)')

    plt.subplot(2,3,3); 
    plt.imshow(LFP_mtxC, aspect='auto', origin='lower', cmap='jet', vmin = -level_max, vmax = level_max); 
    for i in np.arange(0,96,5):
        plt.plot(LFP_mtxC[i,:]*8/level_max + i,'k'); 
    plt.xticks(np.arange(0,1500,250), labels=np.arange(-100,500,100)); 
    plt.yticks(np.arange(-1,96,5), labels=np.arange(20,3860,200));     
    plt.ylim([0, 95]); 
    plt.title('average. Raw LFP');     
    plt.xlabel('Time from stimulus onset (ms)')    
    plt.ylabel('Distance from NPX tip (micrometer)')    


    plt.subplot(2,3,4); 
    plt.imshow(LFP_mtxA2, aspect='auto', origin='lower', cmap='jet', vmin = -level_max2, vmax = level_max2); 
    for i in np.arange(0,95,5):
        plt.plot(LFP_mtxA2[i,:]*8/level_max2 + i,'k'); 
    plt.xticks(np.arange(0,1500,250), labels=np.arange(-100,500,100)); 
    plt.yticks(np.arange(-1,96,5), labels=np.arange(20,3860,200));     
    plt.ylim([0, 95]); 
    plt.title('odd rows. CSD'); 
    plt.xlabel('Time from stimulus onset (ms)')
    plt.ylabel('Distance from NPX tip (micrometer)')

    plt.subplot(2,3,5); 
    plt.imshow(LFP_mtxB2, aspect='auto', origin='lower', cmap='jet', vmin = -level_max2, vmax = level_max2); 
    for i in np.arange(0,95,5):
        plt.plot(LFP_mtxB2[i,:]*8/level_max2 + i,'k'); 
    plt.xticks(np.arange(0,1500,250), labels=np.arange(-100,500,100)); 
    plt.yticks(np.arange(-1,96,5), labels=np.arange(20,3860,200));     
    plt.ylim([0, 95]); 
    plt.title('even rows. CSD');     
    plt.xlabel('Time from stimulus onset (ms)')    
    plt.ylabel('Distance from NPX tip (micrometer)')    

    plt.subplot(2,3,6); 
    plt.imshow(LFP_mtxC2, aspect='auto', origin='lower', cmap='jet', vmin = -level_max2, vmax = level_max2); 
    for i in np.arange(0,95,5):
        plt.plot(LFP_mtxC2[i,:]*8/level_max2 + i,'k'); 
    plt.xticks(np.arange(0,1500,250), labels=np.arange(-100,500,100)); 
    plt.yticks(np.arange(-1,96,5), labels=np.arange(20,3860,200));     
    plt.ylim([0, 95]); 
    plt.title('average. CSD');     
    plt.xlabel('Time from stimulus onset (ms)')    
    plt.ylabel('Distance from NPX tip (micrometer)')    

    plt.tight_layout(); 
    plt.show()

    ### save experiment (processed file)
    experiment = dict(); 
    experiment['filename'] = dat_filename; 
    experiment['NPX_chpos'] = NPX_chpos; 
    experiment['LFP_mtx'] = LFP_mtx;    


    path_to_save = imec_filename[:(imec_filename.rfind('/')+1)] + 'processed/'; 
    if os.path.exists(path_to_save)==0:
        os.mkdir(path_to_save); 
    name_to_save = path_to_save + bin_filename[(bin_filename.rfind('/')+1):-8] + 'npz';
    np.savez_compressed(name_to_save, **experiment); 

    print('processed file was saved');     
    """
    name_to_save = path_to_save + bin_filename[(bin_filename.rfind('/')+1):-8] + 'json.gz';
    f = gzip.GzipFile(name_to_save,'w');    
    f.write(json.dumps(experiment, cls=NumpyEncoder).encode('utf-8')); 
    f.close(); 
    print('processed file was saved'); 
    """


def get_NPX_chpos(NPX_type='3B1_staggered'):
    ### channel map information from https://github.com/cortex-lab/neuropixels

    if NPX_type=='3B1_staggered':
        NPX_chpos = np.empty((384,2)); 
        ### xcoords        
        NPX_chpos[np.arange(0,384,4),0] = 43; 
        NPX_chpos[np.arange(1,384,4),0] = 11; 
        NPX_chpos[np.arange(2,384,4),0] = 59; 
        NPX_chpos[np.arange(3,384,4),0] = 27;                         

        ### ycoords
        NPX_chpos[np.arange(0,384,2),1] = np.arange(20,3860,20); 
        NPX_chpos[np.arange(1,384,2),1] = np.arange(20,3860,20);                 

    return NPX_chpos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='bandpass', output='sos')

    y = sosfilt(sos, data)
    return y




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


#%%
"""
level_max = np.max([np.abs(LFP_mtxA), np.abs(LFP_mtxB), np.abs(LFP_mtxC)])*0.8; 
level_max2 = np.max([np.abs(LFP_mtxA2), np.abs(LFP_mtxB2), np.abs(LFP_mtxC2)])*0.6; 


plt.figure(figsize=(6,3))
plt.subplot(1,2,1); 
plt.imshow(LFP_mtxC, aspect='auto', origin='lower', cmap='jet', vmin = -level_max, vmax = level_max); 
for i in np.arange(0,96,5):
    plt.plot(LFP_mtxC[i,:]*10/level_max + i,'k'); 
plt.xticks(np.arange(0,1500,250), labels=np.arange(-100,500,100)); 
plt.yticks(np.arange(-1,96,5), labels=np.arange(0,3860,200));     
plt.ylim([0, 88]); 
plt.title('average. Raw LFP');     
plt.xlabel('Time from stimulus onset (ms)')    
plt.ylabel('Distance from NPX tip (micrometer)')    

plt.subplot(1,2,2)
plt.imshow(LFP_mtxC2, aspect='auto', origin='lower', cmap='jet', vmin = -level_max2, vmax = level_max2); 
for i in np.arange(0,95,5):
    plt.plot(LFP_mtxC2[i,:]*8/level_max2 + i,'k'); 
plt.xticks(np.arange(0,1500,250), labels=np.arange(-100,500,100)); 
plt.yticks(np.arange(-1,96,5), labels=np.arange(0,3860,200));     
plt.ylim([0, 88]); 
plt.title('average. CSD');     
plt.xlabel('Time from stimulus onset (ms)')    
plt.ylabel('Distance from NPX tip (micrometer)')    

plt.tight_layout(); 
plt.savefig(file_dir+'CSD_example.pdf')
"""
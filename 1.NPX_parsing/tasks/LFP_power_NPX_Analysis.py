#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:38:14 2019

ClutterStim_NPX_Analysis.py

@author: taekjunkim
"""
#%%
import matplotlib.pyplot as plt;
import scipy.optimize as opt;
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfilt, spectrogram, welch

#import sys;
#sys.path.append('./helper'); 

import numpy as np;
import glob; 

import os; 
from helper import parseTJexperiment_NPX as parse_NPX;
import json;
import gzip;

#%%
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
    syncONs = np.where(rawData[syncCh,:imSampRate*20]
                      >np.max(rawData[syncCh,:imSampRate*20])*0.5)[0];    
    for p in range(10):
        if syncONs[p+10]-syncONs[p]==10:
            syncON = syncONs[p]; 
            break; 
    
    ### LFP cut, align at pdON
    LFP_mtx = np.empty((384,int(imSampRate*0.5),len(pdOnTS))); 
    LFP_TS = (np.arange(np.shape(rawData)[1]) - syncON)/imSampRate; 
    for i in range(len(pdOnTS)):
        tsNow = np.where((LFP_TS>=pdOnTS[i]-0.1) & (LFP_TS<=pdOnTS[i]+0.4))[0]; 
        tsNow = tsNow[:int(0.5*imSampRate)]; 

        LFP_now = rawData[:384, tsNow]; 

        ### subtract baseline
        LFP_now = LFP_now - np.mean(LFP_now[:,:int(0.1*imSampRate)],axis=1).reshape(384,1); 

        ### apply butter bandpass filter
        LFP_now = butter_bandpass_filter(LFP_now, lowcut=0.3, highcut=250, fs=imSampRate, order=4); 
        LFP_mtx[:,:,i] = LFP_now; 

    ### Fix ch 191
    LFP_mtx[191,:,:] = (LFP_mtx[190,:,:]+LFP_mtx[192,:,:])/2; 
    LFP_mtx1 = np.mean(LFP_mtx,axis=2).squeeze();     

    ### get_NPX_chpos 
    NPX_chpos = get_NPX_chpos('3B1_staggered'); 
    #NPX_chpos = get_NPX_chpos('3B2_aligned'); 

    ###########
    ### CSD ###
    ###########

    ### divide LFP_mtx into 2 groups based on ycoords
    LFP_mtx1A = LFP_mtx1[np.where(NPX_chpos[:,0]>=43)[0],:];  
    LFP_mtx1B = LFP_mtx1[np.where(NPX_chpos[:,0]<=27)[0],:];  
    LFP_mtx1C = (LFP_mtx1A + LFP_mtx1B)/2; 

    ### denoise LFP_mtx with gaussian_filter
    LFP_mtx1C = gaussian_filter1d(LFP_mtx1C, sigma=5, axis=0, mode='reflect'); 

    ### compute CSD. negative of the 2nd spatial derivative
    spacing_mm = 0.02; 
    LFP_mtx1C2 = -np.diff(LFP_mtx1C, n=2, axis=0)/spacing_mm**2;     

    level_max = np.max([np.abs(LFP_mtx1C)]); 
    level_max2 = np.max([np.abs(LFP_mtx1C2)])*0.6; 

    ### draw figures; 
    plt.figure(figsize=(6,6)); 

    plt.subplot(2,2,1); 
    plt.imshow(LFP_mtx1C, aspect='auto', origin='lower', cmap='jet', vmin = -level_max, vmax = level_max); 
    for i in np.arange(0,192,10):
        plt.plot(LFP_mtx1C[i,:]*8/level_max + i,'k'); 
    plt.xticks(np.arange(0,1500,250), labels=np.arange(-100,500,100)); 
    plt.yticks(np.arange(-1,192,10), labels=np.arange(20,3860,200));     
    plt.ylim([0, 191]); 
    plt.title('average. Raw LFP');     
    plt.xlabel('Time from stimulus onset (ms)')    
    plt.ylabel('Distance from NPX tip (micrometer)')    

    plt.subplot(2,2,3); 
    plt.imshow(LFP_mtx1C2, aspect='auto', origin='lower', cmap='jet', vmin = -level_max2, vmax = level_max2); 
    for i in np.arange(0,190,10):
        plt.plot(LFP_mtx1C2[i,:]*8/level_max2 + i,'k'); 
    plt.xticks(np.arange(0,1500,250), labels=np.arange(-100,500,100)); 
    plt.yticks(np.arange(-1,192,10), labels=np.arange(20,3860,200));     
    plt.ylim([0, 190]); 
    plt.title('average. CSD');     
    plt.xlabel('Time from stimulus onset (ms)')    
    plt.ylabel('Distance from NPX tip (micrometer)')    

    ###############
    ### Spectra ###
    ###############
    LFP_mtx2 = np.empty((384,31,len(pdOnTS)));    # freq: np.arange(0,151,5); 
    for ch in np.arange(384):
        for i in range(len(pdOnTS)):
            sigNow = LFP_mtx[ch,:,i].squeeze(); 
            f,pxx = welch(sigNow,fs=imSampRate,nperseg=500); 
            LFP_mtx2[ch,:,i] = pxx[:31]; 
        print(f'ch#{ch} was done'); 


    LFP_mtx2A = LFP_mtx2[np.where(NPX_chpos[:,0]>=43)[0],:,:];  
    LFP_mtx2B = LFP_mtx2[np.where(NPX_chpos[:,0]<=27)[0],:,:];  
    LFP_mtx2C = (LFP_mtx2A + LFP_mtx2B)/2; 

    plt.subplot(2,2,2); 
    plt.imshow(np.mean(LFP_mtx2C,axis=2),aspect='auto',origin='lower')
    plt.xticks(np.arange(0,31,5),np.arange(0,155,25))
    plt.yticks(np.arange(-1,192,10), labels=np.arange(20,3860,200));     
    plt.xlabel('Frequency (Hz)');    
    plt.ylabel('Distance from NPX tip (micrometer)')    
    plt.title('average. Spectral analysis');         


    plt.tight_layout(); 
    plt.show()

    ### save experiment (processed file)
    experiment = dict(); 
    experiment['filename'] = dat_filename; 
    experiment['NPX_chpos'] = NPX_chpos; 
    experiment['LFP_mtx2'] = LFP_mtx2;    


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

    if NPX_type=='3B2_aligned':        
        NPX_chpos = np.empty((384,2)); 
        ### xcoords
        NPX_chpos[np.arange(0,384,2),0] = 43; 
        NPX_chpos[np.arange(1,384,2),0] = 11;                 
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
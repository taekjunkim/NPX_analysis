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
from scipy.signal import butter, sosfilt, periodogram
from scipy import stats; 

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

    ###############
    ### Spectra ###
    ###############
    freq = np.arange(0,151,2);     
    LFP_mtx2 = np.empty((384,len(freq),len(pdOnTS)));    # freq: np.arange(0,151,5); 
    for ch in np.arange(384):
        for i in range(len(pdOnTS)):
            sigNow = LFP_mtx[ch,:,i].squeeze(); 
            f,pxx = periodogram(sigNow,fs=imSampRate); 
            LFP_mtx2[ch,:,i] = pxx[:len(freq)]; 
        print(f'ch#{ch} was done');     
    LFP_mtx2 = np.mean(LFP_mtx2, axis=2);     
  
    ### Remove bad channels from LFP_mtx1, LFP_mtx2; 
    bad_ch = []; 

    # based on flat LFP
    for i in np.arange(384):
        if np.max(np.abs(LFP_mtx1[i,:]))<np.max(np.abs(LFP_mtx1))*0.2:
            bad_ch.append(i); 
        
    # based on spectra
    for i in np.arange(384):
        fpos1 = np.where((freq>30) & (freq<50))[0]; 
        fpos2 = np.where((freq>50) & (freq<70))[0]; 
        if np.nanmean(LFP_mtx2[i,fpos2])>np.nanmean(LFP_mtx2[i,fpos1])*2:
            bad_ch.append(i); 
        if (i>0) and (i<383):
            if ((np.nanmean(LFP_mtx2[i,fpos2])>np.nanmean(LFP_mtx2[i-1,fpos2])*1.5) and 
                (np.nanmean(LFP_mtx2[i,fpos2])>np.nanmean(LFP_mtx2[i+1,fpos2]))): 
                bad_ch.append(i); 
            if ((np.nanmean(LFP_mtx2[i,fpos2])>np.nanmean(LFP_mtx2[i-1,fpos2])) and 
                (np.nanmean(LFP_mtx2[i,fpos2])>np.nanmean(LFP_mtx2[i+1,fpos2])*1.5)): 
                bad_ch.append(i); 
    
    # based on spectra2: too strong power (outlier)
    too_strong = list(np.where(stats.zscore(np.nansum(LFP_mtx2,axis=1))>3)[0]); 
    bad_ch = bad_ch + too_strong; 

    bad_ch = np.unique(np.array(bad_ch)); 

    LFP_mtx1[bad_ch,:] = np.nan; 
    LFP_mtx2[bad_ch,:] = np.nan; 

    # interpolate LFP_mtx1
    for i in np.arange(np.shape(LFP_mtx1)[1]):
        sig = LFP_mtx1[:,i]; 
        xp = np.where(np.isnan(sig)==0)[0]
        fp = sig[xp]
        x  = np.arange(xp[0],xp[-1]); 
        LFP_mtx1[x,i] = np.interp(x, xp, fp)

    # interpolate LFP_mtx2
    for i in np.arange(np.shape(LFP_mtx2)[1]):
        sig = LFP_mtx2[:,i]; 
        xp = np.where(np.isnan(sig)==0)[0]
        fp = sig[xp]
        x  = np.arange(xp[0],xp[-1]); 
        LFP_mtx2[x,i] = np.interp(x, xp, fp)

    ### get_NPX_chpos 
    NPX_chpos = get_NPX_chpos('3B1_staggered'); 
    #NPX_chpos = get_NPX_chpos('3B2_aligned'); 

    ### divide LFP_mtx into 2 groups based on ycoords
    LFP_mtx2A = LFP_mtx2[np.where(NPX_chpos[:,0]>=43)[0],:];  
    LFP_mtx2B = LFP_mtx2[np.where(NPX_chpos[:,0]<=27)[0],:];  

    ### For CSD ###
    ### divide LFP_mtx into 2 groups based on ycoords
    LFP_mtx1A = LFP_mtx1[np.where(NPX_chpos[:,0]>=43)[0],:];  
    LFP_mtx1B = LFP_mtx1[np.where(NPX_chpos[:,0]<=27)[0],:];          

    ### draw figures; 
    plt.figure(figsize=(12,6)); 
    draw_CSD(LFP_mtx1A, colnum=1); 
    draw_CSD(LFP_mtx1B, colnum=2); 
    
    draw_Spectra(LFP_mtx2A, colnum=3); 
    draw_Spectra(LFP_mtx2B, colnum=4); 
    plt.tight_layout(); 
    path_to_save = imec_filename[:(imec_filename.rfind('/')+1)] + 'processed/'; 
    if os.path.exists(path_to_save)==0:
        os.mkdir(path_to_save); 
    fig_name = path_to_save + bin_filename[(bin_filename.rfind('/')+1):-8] + 'pdf';
    plt.savefig(fig_name); 
    plt.show()

    ### save experiment (processed file)
    experiment = dict(); 
    experiment['filename'] = dat_filename; 
    experiment['NPX_chpos'] = NPX_chpos; 
    experiment['LFP_mtx1_CSD'] = LFP_mtx1;    
    experiment['LFP_mtx2_Spectra'] = LFP_mtx2;    

    name_to_save = path_to_save + bin_filename[(bin_filename.rfind('/')+1):-8] + 'npz';
    np.savez_compressed(name_to_save, **experiment); 

    """
    path_to_save = imec_filename[:(imec_filename.rfind('/')+1)] + 'processed/'; 
    if os.path.exists(path_to_save)==0:
        os.mkdir(path_to_save); 
    name_to_save = path_to_save + bin_filename[(bin_filename.rfind('/')+1):-8] + 'npz';
    np.savez_compressed(name_to_save, **experiment); 
    """
    print('processed file was saved');     
    """
    name_to_save = path_to_save + bin_filename[(bin_filename.rfind('/')+1):-8] + 'json.gz';
    f = gzip.GzipFile(name_to_save,'w');    
    f.write(json.dumps(experiment, cls=NumpyEncoder).encode('utf-8')); 
    f.close(); 
    print('processed file was saved'); 
    """

def draw_Spectra(rLFP_mtx, colnum):

    ### denoise LFP_mtx with gaussian_filter
    rLFP_mtx = gaussian_filter1d(rLFP_mtx, sigma=2, axis=0, mode='reflect'); 

    for fq in np.arange(np.shape(rLFP_mtx)[1]):
        rLFP_mtx[:,fq] = rLFP_mtx[:,fq]/np.nanmax(rLFP_mtx[:,fq]); 

    freq = np.arange(0,151,2);     
    alpha_beta = np.where((freq>=10) & (freq<=30))[0]; 
    gamma = np.where((freq>=75) & (freq<=150))[0];    

    plt.subplot(2,4,colnum); 
    plt.imshow(rLFP_mtx,aspect='auto',origin='lower')
    plt.xticks(np.arange(0,76,25),np.arange(0,155,50))
    plt.yticks(np.arange(-1,192,10), labels=np.arange(20,3860,200));     
    plt.plot(alpha_beta,193*np.ones(np.shape(alpha_beta)),'bs'); 
    plt.plot(gamma,193*np.ones(np.shape(gamma)),'rs'); 
    plt.xlabel('Frequency (Hz)');    
    plt.ylabel('Distance from NPX tip (micrometer)')    
    plt.title('average. Spectral analysis');         


    plt.subplot(2,4,colnum+4); 
    plt.plot(np.mean(rLFP_mtx[:,alpha_beta],axis=1),np.arange(0,192),'b',label='alpha-beta'); 
    plt.plot(np.mean(rLFP_mtx[:,gamma],axis=1),np.arange(0,192),'r',label='gamma'); 
    plt.yticks(np.arange(-1,192,10), labels=np.arange(20,3860,200));    
    plt.gca().spines[['right', 'top']].set_visible(False); 
    plt.legend(); 
    plt.xlabel('Relative power');    


def draw_CSD(LFP_mtx, colnum):
    ### denoise LFP_mtx with gaussian_filter
    LFP_mtx = gaussian_filter1d(LFP_mtx, sigma=2, axis=0, mode='reflect'); 

    ### compute CSD. negative of the 2nd spatial derivative
    spacing_mm = 0.02; 
    LFP_mtx1 = -np.diff(LFP_mtx, n=2, axis=0)/spacing_mm**2;     

    level_max = np.nanmax([np.abs(LFP_mtx)]); 
    level_max1 = np.nanmax([np.abs(LFP_mtx1)])*0.6; 


    plt.subplot(2,4,colnum); 
    plt.imshow(LFP_mtx, aspect='auto', origin='lower', cmap='jet', vmin = -level_max, vmax = level_max); 
    for i in np.arange(0,192,10):
        plt.plot(LFP_mtx[i,:]*8/level_max + i,'k'); 
    plt.xticks(np.arange(0,1251,250), labels=np.arange(-100,401,100)); 
    plt.yticks(np.arange(-1,192,10), labels=np.arange(20,3860,200));     
    plt.ylim([0, 191]); 
    plt.title('average. Raw LFP');     
    plt.xlabel('Time from stimulus onset (ms)')    
    plt.ylabel('Distance from NPX tip (micrometer)')    

    plt.subplot(2,4,colnum+4); 
    plt.imshow(LFP_mtx1, aspect='auto', origin='lower', cmap='jet', vmin = -level_max1, vmax = level_max1); 
    for i in np.arange(0,190,10):
        plt.plot(LFP_mtx1[i,:]*8/level_max1 + i,'k'); 
    plt.xticks(np.arange(0,1251,250), labels=np.arange(-100,401,100)); 
    plt.yticks(np.arange(-1,192,10), labels=np.arange(20,3860,200));     
    plt.ylim([0, 190]); 
    plt.title('average. CSD');     
    plt.xlabel('Time from stimulus onset (ms)')    
    plt.ylabel('Distance from NPX tip (micrometer)')    



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
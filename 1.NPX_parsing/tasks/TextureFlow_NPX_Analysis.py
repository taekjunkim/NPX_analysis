#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:38:14 2019

Texture3Ver_NPX_Analysis.py

@author: taekjunkim
"""

#%%
import matplotlib.pyplot as plt;
import scipy.optimize as opt;

#import sys;
#sys.path.append('./helper'); 

import os
import numpy as np;
import makeSDF;
import glob; 

import parseTJexperiment_NPX as parse_NPX;
import json;
import gzip;

#from sklearn import linear_model
import statsmodels.api as sm
import er_est as er; 

#%%
def main(app):

    #%% This needs to be defined
    dat_filename = app.dat_file; 
    imec_filename = app.imec_file; 
    task_folder = dat_filename[:(dat_filename.rfind('/')+1)]; 
    bin_filename = glob.glob(task_folder+'*.nidq.bin')[0]; 

    task_index_in_combine = int(app.tasknum_lineEdit.text()); 

    #%% get experiment
    prevTime = 0.3; 
    numStims = 81; 
    experiment = parse_NPX.main(bin_filename, dat_filename, prevTime, numStims, imec_filename, app); 
    experiment['StimDur'] = experiment['stimon']; 

    TimeOfInterest = np.arange(int(prevTime*1000+40),int(prevTime*1000+experiment['StimDur']+100+1));
    
    StimResp = [];
    mResp = np.zeros((numStims,experiment['numNeurons'])); 
    psth_mtx = np.zeros((numStims,int(experiment['StimDur'] + 300*2),experiment['numNeurons']));     
    for i in np.arange(len(experiment['stimStructs'])):
        StimResp.append(dict()); 
        StimResp[i]['timeOn'] = experiment['stimStructs'][i]['timeOn']
        StimResp[i]['timeOff'] = experiment['stimStructs'][i]['timeOff']   
        StimResp[i]['pdOn'] = experiment['stimStructs'][i]['pdOn']
        StimResp[i]['pdOff'] = experiment['stimStructs'][i]['pdOff'];       
        StimResp[i]['neurons'] = experiment['stimStructs'][i]['neurons'];        

        for j in np.arange(experiment['numNeurons']):
            NumRepeat = len(StimResp[i]['pdOn'])
            sigLength = int(experiment['StimDur'] + prevTime*1000*2); # to include pre_stim, post_stim
            StimResp[i]['neurons'][j]['spkMtx'] = np.zeros((NumRepeat,sigLength),dtype=int)
            
            for r in np.arange(NumRepeat):
                spkTime = StimResp[i]['neurons'][j]['spikes'][r] - StimResp[i]['pdOn'][r]
                spkTime = spkTime[:]*1000 + prevTime*1000; 
                spkTime = spkTime.astype(int)
                spkTime = spkTime[np.where(spkTime<sigLength)]
                StimResp[i]['neurons'][j]['spkMtx'][r,spkTime] = 1
            
            StimResp[i]['neurons'][j]['meanSDF'] = makeSDF.getSDF(StimResp[i]['neurons'][j]['spkMtx'],1000)
            psth_mtx[i,:,j] = makeSDF.getSDF(StimResp[i]['neurons'][j]['spkMtx'],1000);            
            mResp[i,j] = np.mean(StimResp[i]['neurons'][j]['meanSDF'][TimeOfInterest])

            
    del experiment['stimStructs'];
    del experiment['iti_start'];
    del experiment['iti_end'];
    #del experiment['iti'];
    experiment['filename'] = dat_filename; 
    experiment['StimResp'] = StimResp; 
    experiment['mResp'] = mResp; 
    experiment['psth_mtx'] = psth_mtx; 

    expand_resp = mResp[np.arange(0,80,2),:]; 
    contract_resp = mResp[np.arange(1,80,2),:]; 

    expand_psth = psth_mtx[np.arange(0,80,2),:,:]; 
    contract_psth = psth_mtx[np.arange(1,80,2),:,:]; 

    neurons_from_strong = np.sum(mResp,axis=0).argsort()[::-1]; 

    plt.figure(figsize=(9, 6))
    ax1 = plt.subplot(2,3,1); 
    ax2 = plt.subplot(2,3,2); 
    ax3 = plt.subplot(2,3,4);     
    ax4 = plt.subplot(2,3,5);     
    ax5 = plt.subplot(2,3,6);     

    for j in np.arange(experiment['numNeurons']):    

        unit_now = neurons_from_strong[j]; 
        if app.all_radiobutton.isChecked() == True:
            unit_id = experiment['neuronid'][unit_now]; 
        elif app.sua_radiobutton.isChecked() == True:
            unit_id = experiment['id_sua'][unit_now]; 
        elif app.mua_radiobutton.isChecked() == True:
            unit_id = experiment['id_mua'][unit_now]; 

        stim_rank = (expand_resp[:,unit_now]+contract_resp[:,unit_now]).argsort()[::-1]; 
        yMax1 = np.max(mResp[:,unit_now]);  
        yMax2 = np.max(psth_mtx[:,:,unit_now]);  

        ax1.clear(); 
        ax1.imshow(expand_psth[stim_rank,:,unit_now], aspect='auto', vmin=0, vmax=yMax2); 
        ax1.set_xticks(np.arange(300,1800,500)); 
        ax1.set_xticklabels(np.arange(0,1500,500)); 
        ax1.set_title(f'unit#: {unit_id}. Expanding'); 

        ax2.clear(); 
        ax2.imshow(contract_psth[stim_rank,:,unit_now], aspect='auto', vmin=0, vmax=yMax2); 
        ax2.set_xticks(np.arange(300,1800,500)); 
        ax2.set_xticklabels(np.arange(0,1500,500)); 
        ax2.set_title(f'unit#: {unit_id}. Contracting'); 

        ax3.clear(); 
        ax3.plot(expand_resp[:,unit_now],contract_resp[:,unit_now],'k.'); 
        ax3.plot([0,yMax1],[0,yMax1],'r'); 
        ax3.set_xlabel('Expanding Texture Response'); 
        ax3.set_ylabel('Contracting Texture Response'); 
        ax3.spines[['right', 'top']].set_visible(False)        

        ax4.clear(); 
        ax4.plot(np.mean(expand_psth[:,:,unit_now],axis=0), label='expanding'); 
        ax4.plot(np.mean(contract_psth[:,:,unit_now],axis=0), label='contracting'); 
        ax4.legend(); 
        ax4.set_xticks(np.arange(300,1800,500)); 
        ax4.set_xticklabels(np.arange(0,1500,500)); 
        ax4.spines[['right', 'top']].set_visible(False)                

        ax5.clear(); 
        ax5.plot(np.mean(expand_psth[:,:,unit_now],axis=0)-np.mean(contract_psth[:,:,unit_now],axis=0),'g', label='difference'); 
        ax5.legend();         
        ax5.set_xticks(np.arange(300,1800,500)); 
        ax5.set_xticklabels(np.arange(0,1500,500)); 
        ax5.spines[['right', 'top']].set_visible(False)                

        plt.tight_layout();       
        plt.pause(3); 

        if app.running == 0:
            break; 


    ### save experiment (processed file)
    path_to_save = imec_filename[:(imec_filename.rfind('/')+1)] + 'processed/'; 
    if os.path.exists(path_to_save)==0:
        os.mkdir(path_to_save); 
    name_to_save = path_to_save + bin_filename[(bin_filename.rfind('/')+1):-8] + 'npz';
    np.savez_compressed(name_to_save, **experiment); 

    #name_to_save = path_to_save + bin_filename[(bin_filename.rfind('/')+1):-8] + 'json.gz';
    #f = gzip.GzipFile(name_to_save,'w');    
    #f.write(json.dumps(experiment, cls=NumpyEncoder).encode('utf-8')); 
    #f.close(); 
    print('processed file was saved'); 

    plt.show(block=True);





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

"""
savedir = '../../../2.DataFiles/ProcessedFiles/RFmapping/'
f = gzip.GzipFile(savedir+filename[:-4]+'.json.gz','w');
f.write(json.dumps(experiment, cls=NumpyEncoder).encode('utf-8'));
f.close();
"""

#%%
#%% load cell data
"""
import json;
import gzip;

savedir = '../../../2.DataFiles/ProcessedFiles/RFmapping/'
f = gzip.GzipFile(savedir+filename[:-4]+'.json.gz','r');
Data = json.loads(f.read().decode('utf-8'));
f.close();
"""
       
#%%


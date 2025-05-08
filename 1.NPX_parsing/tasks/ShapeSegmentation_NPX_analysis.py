#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 6 12:22:14 2024

ShapeSegmentation_NPX_Analysis.py

208 "shape against texture" stims were presented at V4 RF. 
Animal did a simple fixation task while visual stimuli were presented. 

@author: taekjunkim
"""
#%%
import matplotlib.pyplot as plt;

import sys;
sys.path.append('./helper'); 

import os
import numpy as np;
from helper import makeSDF;
import glob; 

from helper import parseTJexperiment_NPX as parse_NPX;
import json;
import gzip;
import pandas as pd; 

#from sklearn import linear_model
from scipy import stats
import statsmodels.api as sm
#import er_est as er; 

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
    numStims = 209;     """(50 stims + 1 blank) x 4 fixations """
    experiment = parse_NPX.main(bin_filename, dat_filename, prevTime, numStims, imec_filename, app, id_from_one=False); 
    experiment['StimDur'] = experiment['stimon']; 

    TimeOfInterest = np.arange(int(prevTime*1000),int(prevTime*1000+experiment['StimDur']+100+1));
    
    StimResp = []; 
    mResp = np.zeros((numStims,experiment['numNeurons']));
    for i in np.arange(len(experiment['stimStructs'])):
        StimResp.append(dict()); 

        NumRepeat = len(experiment['stimStructs'][i]['timeOn']); 
        for r in np.arange(len(experiment['stimStructs'][i]['timeOn'])-1):
            if experiment['stimStructs'][i]['timeOn'][r]>experiment['stimStructs'][i]['timeOn'][r+1]:
                NumRepeat = r+1; 
                break;  

        StimResp[i]['timeOn'] = experiment['stimStructs'][i]['timeOn'][:NumRepeat]
        StimResp[i]['timeOff'] = experiment['stimStructs'][i]['timeOff'][:NumRepeat]   
        StimResp[i]['pdOn'] = experiment['stimStructs'][i]['pdOn'][:NumRepeat]
        StimResp[i]['pdOff'] = experiment['stimStructs'][i]['pdOff'][:NumRepeat]       
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
            mResp[i,j] = np.mean(StimResp[i]['neurons'][j]['meanSDF'][TimeOfInterest])
        print(f'condition {i}/208 was done')    

            
    del experiment['stimStructs'];
    del experiment['iti_start'];
    del experiment['iti_end'];
    #del experiment['iti'];
    experiment['filename'] = dat_filename;
    experiment['StimResp'] = StimResp;
    experiment['mResp'] = mResp;    

    ### make StimTable
    stim_df = pd.DataFrame(columns=['id','shapeID','texID','ori','PS']); 
    stim_df['id'] = np.arange(208); 
    stim_df['shapeID'] = np.nan; 
    stim_df['texID'] = np.nan; 
    stim_df['ori'] = np.nan; 
    stim_df['PS'] = np.nan; 

    stim_df.loc[np.arange(8), 'shapeID'] = np.arange(8); 
    stim_df.loc[np.arange(8,20), 'texID'] = np.arange(12); 
    for texID, i in enumerate(np.arange(20,101,8)):
        stim_df.loc[np.arange(i, (i+8)), 'shapeID'] = np.arange(8); 
        stim_df.loc[np.arange(i, (i+8)), 'texID'] = texID; 
    for texID in np.arange(8): # because I have 8 directional textures
        rowNow = np.where(stim_df['texID']==texID)[0]; 
        stim_df.loc[rowNow, 'ori'] = 45*(texID%4); 
    stim_df.loc[np.arange(108,208), 'PS'] = 1; 
    stim_df.loc[np.arange(108,208), 'shapeID'] = stim_df.loc[np.arange(8,108), 'shapeID'].values;
    stim_df.loc[np.arange(108,208), 'texID'] = stim_df.loc[np.arange(8,108), 'texID'].values; 
    stim_df.loc[np.arange(108,208), 'ori'] = stim_df.loc[np.arange(8,108), 'ori'].values; 

    ### save experiment (processed file)
    path_to_save = imec_filename[:(imec_filename.rfind('/')+1)] + 'processed/'; 
    if os.path.exists(path_to_save)==0:
        os.mkdir(path_to_save); 
    name_to_save = path_to_save + bin_filename[(bin_filename.rfind('/')+1):-8] + 'npz'; 
    np.savez_compressed(name_to_save, **experiment); 

    """
    name_to_save = path_to_save + bin_filename[(bin_filename.rfind('/')+1):-8] + 'json.gz';
    f = gzip.GzipFile(name_to_save,'w');    
    f.write(json.dumps(experiment, cls=NumpyEncoder).encode('utf-8')); 
    f.close(); 
    """
    print('processed file was saved'); 



    plt.figure(figsize=(6, 5)); 
    ax1 = plt.subplot(2,2,1); 
    ax2 = plt.subplot(2,2,2); 
    ax3 = plt.subplot(2,2,3); 
    ax4 = plt.subplot(2,2,4); 

    # sort neurons according to response level (strong to weak)
    neurons_from_strong = np.sum(mResp,axis=0).argsort()[::-1]; 

    for j in np.arange(experiment['numNeurons']):
        unit_now = neurons_from_strong[j]; 
        if app.all_radiobutton.isChecked() == True:
            unit_id = experiment['neuronid'][unit_now]; 
        elif app.sua_radiobutton.isChecked() == True:
            unit_id = experiment['id_sua'][unit_now]; 
        elif app.mua_radiobutton.isChecked() == True:
            unit_id = experiment['id_mua'][unit_now]; 

        respVec = mResp[:, unit_now]; 
        tAlone = respVec[np.arange(8)]; 
        bAlone = respVec[np.arange(8,19)]; 
        shape_rank = tAlone.argsort()[::-1]; 
        txt_rank = bAlone.argsort()[::-1]; 
    
        yMax = np.max(respVec); 

        nonPS_mtx = respVec[np.arange(20,108)].reshape((-1,8)); 
        PS_mtx = respVec[np.arange(120,208)].reshape((-1,8)); 

        nonPS_mtx = nonPS_mtx[txt_rank][:,shape_rank]; 
        PS_mtx = PS_mtx[txt_rank][:,shape_rank]; 
        

    
        ax1.clear(); 
        ax1.imshow(nonPS_mtx/yMax, vmin=0, vmax=1, cmap='binary'); 
        ax1.set_title('Original'); 
        ax1.set_xlabel('Shape rank');   ax1.set_ylabel('Texture rank'); 
        ax2.clear(); 
        ax2.imshow(PS_mtx/yMax, vmin=0, vmax=1, cmap='binary'); 
        ax2.set_title('PS statistics'); 
        ax2.set_xlabel('Shape rank');   ax2.set_ylabel('Texture rank'); 

        ax3.clear(); 
        ax3.plot(tAlone[shape_rank], 'ro-', label='target alone'); 
        ax3.plot(np.mean(nonPS_mtx, axis=0), 'ko-', label='original'); 
        ax3.plot(np.mean(PS_mtx, axis=0), 'o-', color=[0.6, 0.6, 0.6], label='PS'); 
        ax3.set_xlabel('Shape rank'); 
        ax3.legend(); 

        ax4.clear(); 
        ax4.plot(bAlone[txt_rank], 'ro-', label='target alone'); 
        ax4.plot(np.mean(nonPS_mtx, axis=1), 'ko-', label='original'); 
        ax4.plot(np.mean(PS_mtx, axis=1), 'o-', color=[0.6, 0.6, 0.6], label='PS'); 
        ax4.set_xlabel('Texture rank'); 
        ax4.legend(); 


        plt.tight_layout();       
        plt.pause(1); 





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
def draw_figure(axisNow, posA, posB, unit, mResp): 
    axisNow.clear()
    x_val = mResp[51*(posA-1):51*posA, unit]; 
    y_val = mResp[51*(posB-1):51*posB, unit];     
    maxResp = np.max(mResp[:,unit])

    axisNow.plot()
    axisNow.plot(x_val, y_val,'k.'); 
    axisNow.plot([0, maxResp],[0, maxResp], 'r'); 
    axisNow.set_xlabel(f'Responses at pos{posA}'); 
    axisNow.set_ylabel(f'Responses at pos{posB}'); 

    rval, pval = stats.pearsonr(x_val,y_val); 
    axisNow.set_title(f'r = {round(rval,2)}. p = {round(pval,2)}'); 

    x_input = sm.add_constant(x_val)
    regr = sm.OLS(y_val,x_input).fit(); 
    y_pred = regr.predict(x_input);
    axisNow.plot(x_val, y_pred, 'g', linewidth=2);  



    return axisNow


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

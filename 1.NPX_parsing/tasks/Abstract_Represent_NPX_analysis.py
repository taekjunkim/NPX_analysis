#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 9 2025

Abstract_Represent_NPX_Analysis.py

160 stims were presented at V4 RF. 
Animal did a simple fixation task while visual stimuli were presented. 

@author: taekjunkim
"""
#%%
import matplotlib.pyplot as plt;

#import sys;
#sys.path.append('./helper'); 

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

    #%% check stimulus conditions
    png_folder = '/Users/taekjunkim/Documents/UW_PasupathyLab/Experiments/11.AbstractRepresentation/png_files/'
    png_files = os.listdir(png_folder)
    png_files.sort()   # to load png images with the right sequence

    stimTable_df = pd.DataFrame(columns=['id','filename','objID','posID','facing','abstraction']); 
    pos_list = np.array(['N30','N45','N60','N90','P30','P45','P60','P90']); 
    for f in np.arange(len(png_files)):
        stimTable_df.at[f,'id'] = f; 
        stimTable_df.at[f,'filename'] = png_files[f]; 
        stimTable_df.at[f,'objID'] = png_files[f][:5]; 
        stimTable_df.at[f,'posID'] = np.where(pos_list==png_files[f][6:9])[0][0]; 
        if png_files[f][6]=='N':
            stimTable_df.at[f,'facing'] = 'L'; 
        else:
            stimTable_df.at[f,'facing'] = 'R'; 
        stimTable_df.at[f,'abstraction'] = png_files[f][10:-4]; 
    stimTable_df.loc[len(png_files),:] = [len(png_files), np.nan, np.nan, np.nan, np.nan, np.nan];   # Blank

    #%% get experiment
    prevTime = 0.3; 
    numStims = 161;     """ 160 stims + 1 blank """
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

        StimResp[i]['id'] = stimTable_df.at[i, 'id']; 
        StimResp[i]['filename'] = stimTable_df.at[i, 'filename']; 
        StimResp[i]['objID'] = stimTable_df.at[i, 'objID']; 
        StimResp[i]['posID'] = stimTable_df.at[i, 'posID']; 
        StimResp[i]['facing'] = stimTable_df.at[i, 'facing']; 
        StimResp[i]['abstraction'] = stimTable_df.at[i, 'abstraction']; 

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
        print(f'condition {i}/508 was done')    

            
    del experiment['stimStructs'];
    del experiment['iti_start'];
    del experiment['iti_end'];
    #del experiment['iti'];
    experiment['filename'] = dat_filename;
    experiment['StimResp'] = StimResp; 
    experiment['mResp'] = mResp;    


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




#%% load cell data
"""
import json;
import gzip;

savedir = '../../../2.DataFiles/ProcessedFiles/RFmapping/'
f = gzip.GzipFile(savedir+filename[:-4]+'.json.gz','r');
Data = json.loads(f.read().decode('utf-8'));
f.close();
"""



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

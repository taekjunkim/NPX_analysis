#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:38:14 2019

KianiGaze_NPX_Analysis.py

50 Kiani objects were presented at V4 RF. 
Animal did a simple fixation task while visual stimuli were presented. 
There were 4 fixation locations (i.e., different gaze positions). 
We were interested in whether response selectivity/magnitude is affected by gaze positions. 

@author: taekjunkim
"""
#%%
import matplotlib.pyplot as plt;

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
    numStims = 111;     """(10 Shape x 11 Texture) + 1 blank """
    experiment = parse_NPX.main(bin_filename, dat_filename, prevTime, numStims, imec_filename, app); 
    experiment['StimDur'] = experiment['stimon']; 

    TimeOfInterest = np.arange(int(prevTime*1000+50),int(prevTime*1000+experiment['StimDur']+100+1));
    
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
        print(f'condition {i}/111 was done')    

            
    del experiment['stimStructs'];
    del experiment['iti_start'];
    del experiment['iti_end'];
    #del experiment['iti'];
    experiment['filename'] = dat_filename;
    experiment['StimResp'] = StimResp;
    experiment['mResp'] = mResp;    

    ### save experiment (processed file)
    #path_to_save = imec_filename[:(imec_filename.rfind('/')+1)] + 'processed/'; 
    #if os.path.exists(path_to_save)==0:
    #    os.mkdir(path_to_save); 
    #name_to_save = path_to_save + bin_filename[(bin_filename.rfind('/')+1):-8] + 'npz';
    #np.savez_compressed(name_to_save, **experiment); 

    #name_to_save = path_to_save + bin_filename[(bin_filename.rfind('/')+1):-8] + 'json.gz';
    #f = gzip.GzipFile(name_to_save,'w');    
    #f.write(json.dumps(experiment, cls=NumpyEncoder).encode('utf-8')); 
    #f.close(); 
    #print('processed file was saved'); 

    neurons_from_strong = np.sum(mResp,axis=0).argsort()[::-1]; 
    nClusters = experiment['numNeurons']; 

    ### drawing part
    plt.figure(figsize=(9,10)); 
    ax1 = plt.subplot(2,2,1); 
    ax2 = plt.subplot(2,2,2); 
    ax3 = plt.subplot(2,2,3); 
    ax4 = plt.subplot(2,2,4); 


    for jj in np.arange(experiment['numNeurons']):
        j = neurons_from_strong[jj]; 
        if app.all_radiobutton.isChecked() == True:
            unit_id = experiment['neuronid'][j]; 
        elif app.sua_radiobutton.isChecked() == True:
            unit_id = experiment['id_sua'][j]; 
        elif app.mua_radiobutton.isChecked() == True:
            unit_id = experiment['id_mua'][j]; 

        respNow = mResp[:110,j]; 
        respNow = np.reshape(respNow,(10,11)); 
    
        shapeResp = np.mean(respNow,axis=1); 
        texResp = np.mean(respNow,axis=0); 

        s_order = shapeResp.argsort()[::-1]; 
        t_order = texResp.argsort()[::-1]; 

        #grayShapes = respNow[1:,0]; 
        #circleTextures = respNow[0,1:]; 
    
        #s_order = grayShapes.argsort()[::-1]+1; 
        #t_order = circleTextures.argsort()[::-1]+1; 
    
        # sort by shape preference
        #respMtx = respNow[np.concatenate((np.array([0]),s_order)),:]; 
        respMtx = respNow[s_order,:]; 
        # sort by texture preference
        #respMtx = respMtx[:,np.concatenate((np.array([0]),t_order))]; 
        respMtx = respMtx[:,t_order]; 

        plt.gcf().suptitle(f'Unit {unit_id}', fontsize=16); 

        ax1.clear(); 
        ax1.imshow(respMtx, origin='lower'); 
        ax1.set_xlabel('Texture IDs'); 
        ax1.set_ylabel('Shape IDs'); 
        ax3.set_title('Response Matrix')        
    
        ax2.clear(); 
        ax2.plot(respMtx[:,1:],np.arange(10));                 
        ax2.plot(np.mean(respMtx,axis=1),np.arange(10),'k',linewidth=3); 
        ax2.set_ylabel('Shape preference'); 

        ax3.clear(); 
        ax3.plot(np.arange(11),respMtx.T[:,1:]);         
        ax3.plot(np.arange(11),np.mean(respMtx.T,axis=1),'k',linewidth=3); 
        ax3.set_xlabel('Texture preference'); 
    
        ax4.clear(); 
        reconMtx = np.matmul(np.mean(respMtx,axis=1).reshape(-1,1),
                             np.mean(respMtx,axis=0).reshape(1,-1)); 
        ax4.imshow(reconMtx, origin='lower'); 
        ax4.set_title('Resp reconstructed')

        plt.tight_layout();       
        plt.pause(10); 

        if app.running == 0:
            break; 
    plt.show(block=True);

    """
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
    """

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

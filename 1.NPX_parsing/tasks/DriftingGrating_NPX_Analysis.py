#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:38:14 2019

DriftingGrating_NPX_Analysis.py

@author: taekjunkim
"""
#%%
import matplotlib.pyplot as plt;
import scipy.optimize as opt;

#import sys;
#sys.path.append('./helper'); 

import numpy as np;
import makeSDF;
import glob; 
import pandas as pd; 
import os; 

import parseTJexperiment_NPX as parse_NPX;
import json;
import gzip;


def main(app):

    #%% This needs to be defined
    dat_filename = app.dat_file; 
    imec_filename = app.imec_file; 
    task_folder = dat_filename[:(dat_filename.rfind('/')+1)]; 
    bin_filename = glob.glob(task_folder+'*.nidq.bin')[0]; 
    csv_filename = dat_filename[:-4]+'_stimTable.csv'; 

    task_index_in_combine = int(app.tasknum_lineEdit.text()); 

    #%% load stim_table
    stim_table = pd.read_csv(csv_filename, index_col=0); 

    #%%
    prevTime = 0.3; 
    numConds = int(stim_table['id'].max());  

    experiment = parse_NPX.main(bin_filename, dat_filename, prevTime, numConds, imec_filename, app); 

    experiment['StimDur'] = experiment['stimon']; 

    TimeOfInterest = np.arange(int(prevTime*1000+50),int(prevTime*1000+experiment['StimDur']+100+1));

    #%%
    StimResp = []; 
    mResp = np.zeros((numConds,experiment['numNeurons'])); 
    for i in np.arange(len(experiment['stimStructs'])):
        StimResp.append(dict());
        StimResp[i]['timeOn'] = experiment['stimStructs'][i]['timeOn']; 
        StimResp[i]['timeOff'] = experiment['stimStructs'][i]['timeOff'];    
        StimResp[i]['pdOn'] = experiment['stimStructs'][i]['pdOn']; 
        StimResp[i]['pdOff'] = experiment['stimStructs'][i]['pdOff'];        
        StimResp[i]['neurons'] = experiment['stimStructs'][i]['neurons'];        

        StimResp[i]['Ori'] = stim_table['ori'][i]; 
        StimResp[i]['SF'] = stim_table['sf'][i]; 
        StimResp[i]['TF'] = stim_table['tf'][i]; 

        for j in np.arange(experiment['numNeurons']):
            NumRepeat = len(StimResp[i]['pdOn']);
            sigLength = int(experiment['StimDur'] + experiment['isi']*2); # to include pre_stim, post_stim
            StimResp[i]['neurons'][j]['spkMtx'] = np.zeros((NumRepeat,sigLength),dtype=int);
            
            for r in np.arange(NumRepeat):
                spkTime = StimResp[i]['neurons'][j]['spikes'][r] - StimResp[i]['pdOn'][r];
                spkTime = spkTime[:]*1000 + experiment['isi'];
                spkTime = spkTime.astype(int);
                spkTime = spkTime[np.where(spkTime<sigLength)];
                StimResp[i]['neurons'][j]['spkMtx'][r,spkTime] = 1;
            
            StimResp[i]['neurons'][j]['meanSDF'] = makeSDF.getSDF(StimResp[i]['neurons'][j]['spkMtx'],1000);
            mResp[i,j] = np.mean(StimResp[i]['neurons'][j]['meanSDF'][TimeOfInterest])
            
    #%%
    del experiment['stimStructs'];
    del experiment['iti_start'];
    del experiment['iti_end'];
    #del experiment['iti'];
    experiment['filename'] = dat_filename; 
    experiment['StimResp'] = StimResp; 

    ### save experiment (processed file)
    path_to_save = imec_filename[:(imec_filename.rfind('/')+1)] + 'processed/'; 
    if os.path.exists(path_to_save)==0:
        os.mkdir(path_to_save); 
    #name_to_save = path_to_save + bin_filename[(bin_filename.rfind('/')+1):-8] + 'npz';
    #np.savez_compressed(name_to_save, **experiment); 

    name_to_save = path_to_save + bin_filename[(bin_filename.rfind('/')+1):-8] + 'json.gz';
    f = gzip.GzipFile(name_to_save,'w');    
    f.write(json.dumps(experiment, cls=NumpyEncoder).encode('utf-8')); 
    f.close(); 
    print('processed file was saved'); 



    ### draw results
    # sort neurons according to response level (strong to weak)
    neurons_from_strong = np.sum(mResp,axis=0).argsort()[::-1]; 

    # get conditions
    SFs = stim_table['sf'].unique(); 
    ORs = stim_table['ori'].unique();  


    plt.figure(figsize=(6, 6))
    ax1 = plt.subplot(2,2,1); 
    ax2 = plt.subplot(2,2,2); 
    ax3 = plt.subplot(2,2,3); 
    ax4 = plt.subplot(2,2,4); 


    nClusters = experiment['numNeurons']; 
    for i in range(nClusters):

        unit_now = neurons_from_strong[i]; 
        if app.all_radiobutton.isChecked() == True:
            unit_id = experiment['neuronid'][unit_now]; 
        elif app.sua_radiobutton.isChecked() == True:
            unit_id = experiment['id_sua'][unit_now]; 
        elif app.mua_radiobutton.isChecked() == True:
            unit_id = experiment['id_mua'][unit_now]; 


        ### Ori-SF (2,2,1)
        ax1.clear()
        for j in range(len(SFs)-1):
            SF_now = SFs[j]; 
            cond_now = np.where(stim_table['sf']==SF_now)[0];  

            if len(cond_now) == (len(ORs)-1):
                ax1.plot(ORs[:-1], mResp[cond_now,unit_now], '-o', label=f'sf={SF_now}'); 
            else:
                pass; 
        if unit_id in experiment['id_sua']:
            ax1.set_title(f'unit_id#: {unit_id} (SUA)');                                                
        elif unit_id in experiment['id_mua']:                    
            ax1.set_title(f'unit_id#: {unit_id} (MUA)');                                                            
        ax1.set_xlabel('Direction (deg)'); 
        ax1.set_ylabel('Response (Hz)'); 
        ax1.legend();     

        ### Ori (2,2,3)
        ax3.clear(); 
        OR_resp = []; 
        for j in range(len(ORs)-1):
            OR_now = ORs[j]; 
            cond_now = np.where(stim_table['ori']==OR_now)[0];  
            OR_resp.append(np.mean(mResp[cond_now,unit_now])); 
        OR_resp = np.array(OR_resp); 
        ax3.plot(ORs[:-1], OR_resp, '-o')
        ax3.set_title('Direction tuning: all SFs averaged')
        ax3.set_xlabel('Direction (deg)');
        ax3.set_ylabel('Response (Hz)');          

        ### SF (2,2,4)
        ax4.clear(); 
        SF_resp = []; 
        for j in range(len(SFs)-1):
            SF_now = SFs[j]; 
            cond_now = np.where(stim_table['sf']==SF_now)[0];  
            SF_resp.append(np.mean(mResp[cond_now,unit_now])); 
        SF_resp = np.array(SF_resp); 
        ax4.plot(SFs[:-1], SF_resp, '-o')
        ax4.set_xscale('log'); 
        ax4.set_title('SF tuning: all ORs averaged')        
        ax4.set_xlabel('Spatial frequency (cyc/deg)');
        ax4.set_ylabel('Response (Hz)');    


        plt.tight_layout(); 
        plt.pause(0.5); 

        if app.running == 0:
            break; 

    plt.show(block=True); 

    name_to_save = path_to_save + bin_filename[(bin_filename.rfind('/')+1):-8] + 'json.gz';
    f = gzip.GzipFile(name_to_save,'w');    
    f.write(json.dumps(experiment, cls=NumpyEncoder).encode('utf-8')); 
    f.close(); 
    print('processed file was saved'); 




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


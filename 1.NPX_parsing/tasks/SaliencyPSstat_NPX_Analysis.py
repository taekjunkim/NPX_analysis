#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:38:14 2019

SaliencyPSstat_NPX_Analysis.py

@author: taekjunkim
"""

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


def main(app):

    #%% This needs to be defined
    dat_filename = app.dat_file; 
    imec_filename = app.imec_file; 
    task_folder = dat_filename[:(dat_filename.rfind('/')+1)]; 
    bin_filename = glob.glob(task_folder+'*.nidq.bin')[0]; 

    task_index_in_combine = int(app.tasknum_lineEdit.text()); 

    #%% get experiment
    prevTime = 0.3; 
    numStims = 125; 
    experiment = parse_NPX.main(bin_filename, dat_filename, prevTime, numStims, imec_filename, app); 
    experiment['StimDur'] = experiment['stimon']; 

    TimeOfInterest = np.arange(int(prevTime*1000+50),int(prevTime*1000+experiment['StimDur']+100+1));
    
    StimResp = [];
    mResp = np.zeros((numStims,experiment['numNeurons']));
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
            mResp[i,j] = np.mean(StimResp[i]['neurons'][j]['meanSDF'][TimeOfInterest])

            
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


    #%% Stimulus conditions: center stim has a texture surface
    """
    lgCenter_alone = np.arange(0,8);   # 0,1,2,3,4,5,6,7
    lgCenter_cirScene = np.arange(8,16);   
    lgCenter_rndScene = np.arange(16,24);   
    mdCenter_alone = np.arange(24,32);
    mdCenter_cirScene = np.arange(32,40);
    mdCenter_rndScene = np.arange(40,48);
    smCenter_alone = np.arange(48,56);
    smCenter_cirScene = np.arange(56,64);
    smCenter_rndScene = np.arange(64,72);
    lgCenter_cirScene_PS = np.arange(72,80);   
    lgCenter_rndScene_PS = np.arange(80,88);   
    mdCenter_cirScene_PS = np.arange(88,96);   
    mdCenter_rndScene_PS = np.arange(96,104);   
    smCenter_cirScene_PS = np.arange(104,112);   
    smCenter_rndScene_PS = np.arange(112,120);   
    cirScene = 120;
    cirScene_PS = 121;
    rndScene = 122;
    rndScene_PS = 123; 
    NoStim = 124; 
    """


#%% Drawing part
    # sort neurons according to response level (strong to weak)
    neurons_from_strong = np.sum(mResp,axis=0).argsort()[::-1]; 

    plt.figure(figsize=(8,16)); 
    ax1 = plt.subplot(4,2,1); 
    ax2 = plt.subplot(4,2,2); 
    ax3 = plt.subplot(4,2,3); 
    ax4 = plt.subplot(4,2,4); 
    ax5 = plt.subplot(4,2,5); 
    ax6 = plt.subplot(4,2,6); 
    ax7 = plt.subplot(4,2,7); 
    ax8 = plt.subplot(4,2,8); 

    nClusters = experiment['numNeurons']; 
    for jj in np.arange(experiment['numNeurons']):
    
        j = neurons_from_strong[jj]; 
        if app.all_radiobutton.isChecked() == True:
            unit_id = experiment['neuronid'][j]; 
        elif app.sua_radiobutton.isChecked() == True:
            unit_id = experiment['id_sua'][j]; 
        elif app.mua_radiobutton.isChecked() == True:
            unit_id = experiment['id_mua'][j]; 
        
        NoStim = np.mean(StimResp[124]['neurons'][j]['meanSDF'][TimeOfInterest]);
        
        ## Control of near surround stimulus number
        lgCenter = mResp[0:8,j]; 
        lgCenter_cirScene = mResp[8:16,j]; 
        lgCenter_rndScene = mResp[16:24,j]; 

        mdCenter = mResp[24:32,j]; 
        mdCenter_cirScene = mResp[32:40,j]; 
        mdCenter_rndScene = mResp[40:48,j]; 

        smCenter = mResp[48:56,j]; 
        smCenter_cirScene = mResp[56:64,j]; 
        smCenter_rndScene = mResp[64:72,j]; 

        lgCenter_cirScene_PS = mResp[72:80,j]; 
        lgCenter_rndScene_PS = mResp[80:88,j]; 

        mdCenter_cirScene_PS = mResp[88:96,j]; 
        mdCenter_rndScene_PS = mResp[96:104,j]; 

        smCenter_cirScene_PS = mResp[104:112,j]; 
        smCenter_rndScene_PS = mResp[112:120,j]; 

        cirScene = mResp[120,j]; 
        cirScene_PS = mResp[121,j]; 
        rndScene = mResp[122,j]; 
        rndScene_PS = mResp[123,j]; 

        ### ranking of shape preference
        rk = np.flip(np.argsort(lgCenter)); ## ranking of shape preference    

        ### Center size (4,2,1); 
        ax1.clear()
        ax1.plot(np.arange(1,9), lgCenter[rk], label='lgCenter', color='C0'); 
        ax1.plot(np.arange(1,9), mdCenter[rk], label='mdCenter', color='C1'); 
        ax1.plot(np.arange(1,9), smCenter[rk], label='smCenter', color='C2'); 
        ax1.set_xlabel('Stim ID')
        ax1.set_ylabel('Responses (spk/s)')
        ax1.legend(); 
        if unit_id in experiment['id_sua']:
            ax1.set_title(f'unit#{unit_id} (SUA): Center Size'); 
        elif unit_id in experiment['id_mua']:    
            ax1.set_title(f'unit#{unit_id} (MUA): Center Size'); 
        
        ### lgCenter 
        ax3.clear()
        ax3.plot(np.arange(1,9), lgCenter[rk], label='lgCenter', color='C0'); 
        ax3.plot(np.arange(1,9), lgCenter_cirScene[rk], label='lgCenter_cirScene', 
                 color='C0', linestyle='--'); 
        ax3.plot(np.arange(1,9), lgCenter_rndScene[rk], label='lgCenter_rndScene', 
                 color='C0', linestyle=':'); 
        ax3.set_xlabel('Stim ID')
        ax3.set_ylabel('Responses (spk/s)')
        ax3.legend(); 
        ax3.set_title('lgCenter'); 

        ### mdCenter - ordered by lgCenter
        ax5.clear()
        ax5.plot(np.arange(1,9), mdCenter[rk], label='mdCenter', color='C1'); 
        ax5.plot(np.arange(1,9), mdCenter_cirScene[rk], label='mdCenter_cirScene', 
                 color='C1', linestyle='--'); 
        ax5.plot(np.arange(1,9), mdCenter_rndScene[rk], label='mdCenter_rndScene', 
                 color='C1', linestyle=':'); 
        ax5.set_xlabel('Stim ID')
        ax5.set_ylabel('Responses (spk/s)')
        ax5.legend(); 
        ax5.set_title('mdCenter (order L)'); 

        ### mdCenter - ordered by mdCenter
        rk_m = np.flip(np.argsort(mdCenter)); ## ranking of shape preference            
        ax6.clear()
        ax6.plot(np.arange(1,9), mdCenter[rk_m], label='mdCenter', color='C1'); 
        ax6.plot(np.arange(1,9), mdCenter_cirScene[rk_m], label='mdCenter_cirScene', 
                 color='C1', linestyle='--'); 
        ax6.plot(np.arange(1,9), mdCenter_rndScene[rk_m], label='mdCenter_rndScene', 
                 color='C1', linestyle=':'); 
        ax6.set_xlabel('Stim ID')
        ax6.set_ylabel('Responses (spk/s)')
        ax6.legend(); 
        ax6.set_title('mdCenter (order M)'); 

        ### smCenter - ordered by lgCenter
        ax7.clear()
        ax7.plot(np.arange(1,9), smCenter[rk], label='smCenter', color='C2'); 
        ax7.plot(np.arange(1,9), smCenter_cirScene[rk], label='smCenter_cirScene', 
                 color='C2', linestyle='--'); 
        ax7.plot(np.arange(1,9), smCenter_rndScene[rk], label='smCenter_rndScene', 
                 color='C2', linestyle=':'); 
        ax7.set_xlabel('Stim ID')
        ax7.set_ylabel('Responses (spk/s)')
        ax7.legend(); 
        ax7.set_title('smCenter (order L)'); 

        ### smCenter - ordered by smCenter
        rk_s = np.flip(np.argsort(smCenter)); ## ranking of shape preference                    

        ax8.clear()
        ax8.plot(np.arange(1,9), smCenter[rk_s], label='smCenter', color='C2'); 
        ax8.plot(np.arange(1,9), smCenter_cirScene[rk_s], label='smCenter_cirScene', 
                 color='C2', linestyle='--'); 
        ax8.plot(np.arange(1,9), smCenter_rndScene[rk_s], label='smCenter_rndScene', 
                 color='C2', linestyle=':'); 
        ax8.set_xlabel('Stim ID')
        ax8.set_ylabel('Responses (spk/s)')
        ax8.legend(); 
        ax8.set_title('smCenter (order S)'); 


        print(f'{jj}/{nClusters}: unit_id = {unit_id}'); 
        plt.tight_layout();       
        plt.pause(0.5); 

        if app.running == 0:
            break; 

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


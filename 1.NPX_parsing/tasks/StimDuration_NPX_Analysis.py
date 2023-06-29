#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:38:14 2019

StimDuration_NPX_Analysis.py

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

import parseStimDur_experiment_NPX as parse_NPX;
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
    numStims = 45;    ## 9 stims (including blank) x 5 durations
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
            sigLength = int(400 + prevTime*1000 + 400); # to include pre_stim, post_stim
            StimResp[i]['neurons'][j]['spkMtx'] = np.zeros((NumRepeat,sigLength),dtype=int)
            
            for r in np.arange(NumRepeat):
                spkTime = StimResp[i]['neurons'][j]['spikes'][r] - StimResp[i]['pdOn'][r]
                spkTime = spkTime[:]*1000 + prevTime*1000; 
                spkTime = spkTime.astype(int)
                spkTime = spkTime[np.where(spkTime<1100)]
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
    shapes_position1 = np.arange(0,500,5); 
    shapes_position2 = np.arange(1,500,5); 
    shapes_position3 = np.arange(2,500,5); 
    shapes_position4 = np.arange(3,500,5); 
    shapes_position5 = np.arange(4,500,5); 

    NoStim = 500; 
    """


#%% Drawing part
    # sort neurons according to response level (strong to weak)
    neurons_from_strong = np.sum(mResp,axis=0).argsort()[::-1]; 

    plt.figure(figsize=(8,16)); 
    ax1 = plt.subplot(5,2,1); 
    ax2 = plt.subplot(5,2,2); 
    ax3 = plt.subplot(5,2,3); 
    ax4 = plt.subplot(5,2,4); 
    ax5 = plt.subplot(5,2,5); 
    ax6 = plt.subplot(5,2,6); 
    ax7 = plt.subplot(5,2,7); 
    ax8 = plt.subplot(5,2,8); 
    ax9 = plt.subplot(5,2,9); 
    ax10 = plt.subplot(5,2,10); 


    nClusters = experiment['numNeurons']; 
    for jj in np.arange(experiment['numNeurons']):
    
        j = neurons_from_strong[jj]; 
        if app.all_radiobutton.isChecked() == True:
            unit_id = experiment['neuronid'][j]; 
        elif app.sua_radiobutton.isChecked() == True:
            unit_id = experiment['id_sua'][j]; 
        elif app.mua_radiobutton.isChecked() == True:
            unit_id = experiment['id_mua'][j]; 
        
        psth_pos1 = np.zeros((0,600)); 
        psth_pos2 = np.zeros((0,600)); 
        psth_pos3 = np.zeros((0,600)); 
        psth_pos4 = np.zeros((0,600));                         
        psth_pos5 = np.zeros((0,600)); 

        for shapes in np.arange(100):
            ### shapes_position1
            sid = shapes*5;
            psth_pos1 = np.vstack((psth_pos1, StimResp[sid]['neurons'][j]['meanSDF'][200:800])); 

            ### shapes_position2
            sid = shapes*5+1;            
            psth_pos2 = np.vstack((psth_pos2, StimResp[sid]['neurons'][j]['meanSDF'][200:800])); 
            
            ### shapes_position3
            sid = shapes*5+2;            
            psth_pos3 = np.vstack((psth_pos3, StimResp[sid]['neurons'][j]['meanSDF'][200:800]));             
            
            ### shapes_position4
            sid = shapes*5+3;            
            psth_pos4 = np.vstack((psth_pos4, StimResp[sid]['neurons'][j]['meanSDF'][200:800]));             
            
            ### shapes_position5
            sid = shapes*5+4;            
            psth_pos5 = np.vstack((psth_pos5, StimResp[sid]['neurons'][j]['meanSDF'][200:800]));             

        yMax = np.max([np.max(psth_pos1),np.max(psth_pos2),np.max(psth_pos3),np.max(psth_pos4),np.max(psth_pos5)]); 

        ax1.clear(); 
        ax1.imshow(psth_pos1,vmin=0,vmax=yMax); 
        ax3.clear(); 
        ax3.imshow(psth_pos3,vmin=0,vmax=yMax); 
        ax5.clear(); 
        ax5.imshow(psth_pos5,vmin=0,vmax=yMax); 
        ax7.clear(); 
        ax7.imshow(psth_pos7,vmin=0,vmax=yMax); 
        ax9.clear(); 
        ax9.imshow(psth_pos9,vmin=0,vmax=yMax); 


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


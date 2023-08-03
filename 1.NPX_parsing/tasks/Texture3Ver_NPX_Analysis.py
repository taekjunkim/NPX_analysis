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
    numStims = 121; 
    experiment = parse_NPX.main(bin_filename, dat_filename, prevTime, numStims, imec_filename, app); 
    experiment['StimDur'] = experiment['stimon']; 

    TimeOfInterest = np.arange(int(prevTime*1000+40),int(prevTime*1000+experiment['StimDur']+100+1));
    
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
    experiment['mResp'] = mResp;

    depth = np.array(experiment['chpos_sua'])[:,1]; 
    depth_order = depth.argsort();   # from the tip to the top
    
    ### r2er_n2n analysis
    # n neurons x t trials x c conditions
    nNeurons = experiment['numNeurons']; 
    nTrials = 10; 
    #nCond = 40;   # original only 
    nCond = 120;  # 3 versions all

    respMtx = np.empty((nNeurons,nTrials,nCond)); 
    for n in range(nNeurons):
        for t in range(nTrials):
            for c in range(nCond):
                respNow = np.sum(StimResp[c]['neurons'][n]['spkMtx'][t,TimeOfInterest])*1000/len(TimeOfInterest); 
                respMtx[n,t,c] = respNow; 
    # ordered by depth
    respMtx = respMtx[depth_order,:,:]; 

    dist_Mtx = np.zeros((nNeurons,nNeurons));
    r2er_Mtx = np.ones((nNeurons,nNeurons));
    r2_Mtx = np.ones((nNeurons,nNeurons));
    for n1 in np.arange(nNeurons-1):
        for n2 in np.arange(n1+1,nNeurons): 
            r2er, r2 = er.r2er_n2n(respMtx[n1,:,:]**0.5,respMtx[n2,:,:]**0.5); 
            if r2er < 0:
                r2er = 0; 
            if r2er > 1:
                r2er = 1; 
            r2er_Mtx[n1,n2] = r2er; 
            r2er_Mtx[n2,n1] = r2er;             
            r2_Mtx[n1,n2] = r2; 
            r2_Mtx[n2,n1] = r2; 

            # dist
            dist = np.abs(depth[depth_order[n1]]-depth[depth_order[n2]]); 
            dist_Mtx[n1,n2] = dist; 
            dist_Mtx[n2,n1] = dist; 


    ### save experiment (processed file)
    path_to_save = imec_filename[:(imec_filename.rfind('/')+1)] + 'processed/'; 
    if os.path.exists(path_to_save)==0:
        os.mkdir(path_to_save); 
    #name_to_save = path_to_save + bin_filename[(bin_filename.rfind('/')+1):-8] + 'npz';
    #np.savez_compressed(name_to_save, **experiment); 

    plt.figure(figsize=(10,6)); 
    plt.subplot(2,3,1); 
    rMtx = np.corrcoef(mResp[:nCond,depth_order],rowvar=False); 
    rMtx[np.isnan(rMtx)] = 0; 

    neg_r = np.where(rMtx<0); 
    plt.imshow(rMtx,origin='lower'); 
    plt.colorbar(fraction=0.046, pad=0.04,label='Correlation coefficient (r)')
    plt.title('TexResp: correlation'); 

    plt.subplot(2,3,4); 
    nSig = []; 
    for i in range(nNeurons):    
        x_data = dist_Mtx[:,i]; 
        x_data[i] = np.nan; 
        y_data = rMtx[:,i]; 
        y_data[i] = np.nan; 
        x_input = x_data[(~np.isnan(x_data) & ~np.isnan(y_data))]; 
        y_input = y_data[(~np.isnan(x_data) & ~np.isnan(y_data))]; 

        x_input2 = sm.add_constant(x_input)
        regr = sm.OLS(y_input,x_input2).fit(); 
        y_pred = regr.predict(x_input2); 
        if (regr.pvalues[1]<0.05) and (regr.params[1]<0) and (regr.params[0]>0):
            plt.plot(x_input, y_pred, color=(0,0,0), linewidth=2); 
            nSig.append(i); 
        else:
            plt.plot(x_input, y_pred, color=(0.75,0.75,0.75), linewidth=1);             
    plt.xlabel('Distance between units (micrometer)'); 
    plt.ylabel('Correlation between responses'); 
    plt.title(f"{len(nSig)} / {nNeurons}"); 

    plt.subplot(2,3,2); 
    r_er_Mtx = r2er_Mtx**0.5; 
    r_er_Mtx[neg_r[0],neg_r[1]] = -1*r_er_Mtx[neg_r[0],neg_r[1]]; 
    plt.imshow(r_er_Mtx,origin='lower'); 
    plt.colorbar(fraction=0.046, pad=0.04,label='Correlation coefficient (r)')
    plt.title("TexResp: r2er"); 

    plt.subplot(2,3,5); 
    nSig = [];     
    for i in range(nNeurons):    
        x_data = dist_Mtx[:,i]; 
        x_data[i] = np.nan; 
        y_data = r_er_Mtx[:,i]; 
        y_data[i] = np.nan;         

        x_input = x_data[(~np.isnan(x_data) & ~np.isnan(y_data))]; 
        y_input = y_data[(~np.isnan(x_data) & ~np.isnan(y_data))]; 

        x_input2 = sm.add_constant(x_input)
        regr = sm.OLS(y_input,x_input2).fit(); 
        y_pred = regr.predict(x_input2); 
        if (regr.pvalues[1]<0.05) and (regr.params[1]<0) and (regr.params[0]>0):
            plt.plot(x_input, y_pred, color=(0,0,0), linewidth=2); 
            nSig.append(i);         
        else:
            plt.plot(x_input, y_pred, color=(0.75,0.75,0.75), linewidth=1);    
    plt.xlabel('Distance between units (micrometer)'); 
    plt.ylabel('Correlation between responses');     
    plt.title(f"{len(nSig)} / {nNeurons}");     

    plt.subplot(2,3,3); 
    r_Mtx = r2_Mtx**0.5; 
    r_Mtx[np.isnan(r_Mtx)] = 0; 

    r_Mtx[neg_r[0],neg_r[1]] = -1*r_Mtx[neg_r[0],neg_r[1]]; 
    plt.imshow(r_Mtx,origin='lower'); 
    plt.colorbar(fraction=0.046, pad=0.04,label='Correlation coefficient (r)')
    plt.title('TexResp: r'); 

    plt.subplot(2,3,6); 
    dist_Mtx2 = dist_Mtx; 
    r_Mtx2 = r_Mtx; 
    for i in range(nNeurons):
        dist_Mtx2[i,i] = np.nan; 
        r_Mtx2[i,i] = np.nan; 
    x_data = dist_Mtx2.ravel(); 
    y_data = r_Mtx2.ravel();     
    x_inputA = x_data[(~np.isnan(x_data) & ~np.isnan(y_data))]; 
    y_inputA = y_data[(~np.isnan(x_data) & ~np.isnan(y_data))]; 
    x_inputB = x_data[(~np.isnan(x_data) & ~np.isnan(y_data)) & (x_data<1500)]; 
    y_inputB = y_data[(~np.isnan(x_data) & ~np.isnan(y_data)) & (x_data<1500)]; 

    x_inputA2 = sm.add_constant(x_inputA)
    x_inputB2 = sm.add_constant(x_inputB)    

    regrA = sm.OLS(y_inputA, x_inputA2).fit(); 
    y_predA = regrA.predict(x_inputA2); 

    regrB = sm.OLS(y_inputB, x_inputB2).fit(); 
    y_predB = regrB.predict(x_inputB2); 

    plt.plot(x_inputA,y_inputA,'k.'); 
    plt.plot(x_inputA, y_predA, color=(0,0,1), linewidth=2); 
    plt.plot(x_inputB, y_predB, color=(1,0,0), linewidth=2);     
    plt.xlabel('Distance between units (micrometer)'); 
    plt.ylabel('Correlation between responses'); 

    plt.tight_layout(); 
    plt.savefig(path_to_save+"TexCluster.pdf", format="pdf", bbox_inches="tight"); 


    name_to_save = path_to_save + bin_filename[(bin_filename.rfind('/')+1):-8] + 'json.gz';
    f = gzip.GzipFile(name_to_save,'w');    
    f.write(json.dumps(experiment, cls=NumpyEncoder).encode('utf-8')); 
    f.close(); 
    print('processed file was saved'); 


    #%% Stimulus conditions: center stim has a texture surface
    """
    OrigTex = np.arange(0,40);
    ContRevTex = np.arange(40,80);   
    NoiseTex = np.arange(80,120);   
    NoStim = 121; 
    """


#%% Drawing part
    """
    # sort neurons according to response level (strong to weak)
    neurons_from_strong = np.sum(mResp,axis=0).argsort()[::-1]; 

    plt.figure(figsize=(15,5)); 
    ax1 = plt.subplot(1,3,1); 
    ax2 = plt.subplot(1,3,2); 
    ax3 = plt.subplot(1,3,3); 

    nClusters = experiment['numNeurons']; 
    for jj in np.arange(experiment['numNeurons']):
    
        j = neurons_from_strong[jj]; 
        if app.all_radiobutton.isChecked() == True:
            unit_id = experiment['neuronid'][j]; 
        elif app.sua_radiobutton.isChecked() == True:
            unit_id = experiment['id_sua'][j]; 
        elif app.mua_radiobutton.isChecked() == True:
            unit_id = experiment['id_mua'][j];         
        
        ## Control of near surround stimulus number
        origTex = mResp[0:40,j]; 
        contTex = mResp[40:80,j]; 
        noiseTex = mResp[80:120,j]; 

        ### origTex vs. contTex (1,3,1); 
        ax1.clear()
        ax1.plot(origTex, contTex, 'ko'); 
        ax1.set_xlabel('Responses to origTex')
        ax1.set_ylabel('Responses to contTex')
        if unit_id in experiment['id_sua']:
            ax1.set_title(f'unit#{unit_id} (SUA): origTex vs. contTex'); 
        elif unit_id in experiment['id_mua']:    
            ax1.set_title(f'unit#{unit_id} (MUA): origTex vs. contTex'); 
        
        ### origTex vs. noiseTex (1,3,2); 
        ax2.clear()
        ax2.plot(origTex, noiseTex, 'ko'); 
        ax2.set_xlabel('Responses to origTex')
        ax2.set_ylabel('Responses to contTex')
        if unit_id in experiment['id_sua']:
            ax2.set_title(f'unit#{unit_id} (SUA): origTex vs. noiseTex'); 
        elif unit_id in experiment['id_mua']:    
            ax2.set_title(f'unit#{unit_id} (MUA): origTex vs. noiseTex'); 

        ### contTex vs. noiseTex (1,3,2); 
        ax3.clear()
        ax3.plot(contTex, noiseTex, 'ko'); 
        ax3.set_xlabel('Responses to origTex')
        ax3.set_ylabel('Responses to contTex')
        if unit_id in experiment['id_sua']:
            ax3.set_title(f'unit#{unit_id} (SUA): contTex vs. noiseTex'); 
        elif unit_id in experiment['id_mua']:    
            ax3.set_title(f'unit#{unit_id} (MUA): contTex vs. noiseTex'); 

        print(f'{jj}/{nClusters}: unit_id = {unit_id}'); 
        plt.tight_layout();       
        plt.pause(0.5); 

        if app.running == 0:
            break; 
    """
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


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:38:14 2019

PositionInvariance_NPX_Analysis.py

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

#from sklearn import linear_model
import statsmodels.api as sm
import er_est as er; 

def main(app):

    #%% This needs to be defined
    dat_filename = app.dat_file; 
    imec_filename = app.imec_file; 
    task_folder = dat_filename[:(dat_filename.rfind('/')+1)]; 
    bin_filename = glob.glob(task_folder+'*.nidq.bin')[0]; 

    task_index_in_combine = int(app.tasknum_lineEdit.text()); 

    #%% get experiment
    prevTime = 0.3; 
    numStims = 251; 
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
    experiment['mResp'] = mResp;    

    if app.sua_radiobutton.isChecked() == True:
        depth = np.array(experiment['chpos_sua'])[:,1]; 
    elif app.mua_radiobutton.isChecked() == True:
        depth = np.array(experiment['chpos_mua'])[:,1]; 
    elif app.all_radiobutton.isChecked() == True:
        depth_sua = np.array(experiment['chpos_sua'])[:,1]; 
        depth_mua = np.array(experiment['chpos_mua'])[:,1]; 
        depth = np.vstack((depth_sua, depth_mua)); 
 
    depth_order = depth.argsort();   # from the tip to the top

    ### r2er_n2n analysis
    # n neurons x t trials x c conditions
    nNeurons = experiment['numNeurons']; 
    nTrials = 10; 
    nCond = 250; 

    respMtxAll = np.empty((nNeurons,nTrials,nCond)); 
    for n in range(nNeurons):
        for t in range(nTrials):
            for c in range(nCond):
                respNow = np.sum(StimResp[c]['neurons'][n]['spkMtx'][t,TimeOfInterest])*1000/len(TimeOfInterest); 
                respMtxAll[n,t,c] = respNow; 

    # response from the best position
    respMtx = np.empty((nNeurons,nTrials,50)); 
    for n in range(nNeurons): 
        posResp = [np.nanmean(mResp[np.arange(0,250,5),n]), 
                   np.nanmean(mResp[np.arange(1,250,5),n]), 
                   np.nanmean(mResp[np.arange(2,250,5),n]),
                   np.nanmean(mResp[np.arange(3,250,5),n]),
                   np.nanmean(mResp[np.arange(4,250,5),n])]; 
        bestPos = np.min(np.where(posResp==np.max(posResp))[0]); 
        respMtx[n,:10,:50] = respMtxAll[n][:,np.arange(bestPos,250,5)]; 
    """
    # consider all positions together
    respMtx = respMtxAll; 
    """
    
    # ordered by depth
    respMtx = respMtx[depth_order,:,:]; 

    dist_Mtx = np.zeros((nNeurons,nNeurons));
    r2er_Mtx = np.ones((nNeurons,nNeurons)); 
    r2_Mtx = np.ones((nNeurons,nNeurons)); 
    for n1 in range(nNeurons-1):
        for n2 in range(n1+1,nNeurons): 
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
    #rMtx = np.corrcoef(mResp[:,depth_order],rowvar=False); 
    rMtx = np.corrcoef(np.nanmean(respMtx,axis=1).squeeze().T,rowvar=False); 
    rMtx[np.isnan(rMtx)] = 0; 

    neg_r = np.where(rMtx<0); 
    plt.imshow(rMtx,origin='lower',cmap='bwr',vmin=-1,vmax=1);  
    plt.colorbar(fraction=0.046, pad=0.04,label='Correlation coefficient (r)')
    plt.title('ShapeResp: correlation'); 

    plt.subplot(2,3,4); 
    nSig = []
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
            nSig.append(i)
        else:
            plt.plot(x_input, y_pred, color=(0.75,0.75,0.75), linewidth=1);             
    plt.xlabel('Distance between units (micrometer)'); 
    plt.ylabel('Correlation between responses'); 
    plt.title(f"{len(nSig)} / {nNeurons}")

    plt.subplot(2,3,2); 
    r_er_Mtx = r2er_Mtx**0.5; 
    r_er_Mtx[np.isnan(r_er_Mtx)] = 0; 

    r_er_Mtx[neg_r[0],neg_r[1]] = -1*r_er_Mtx[neg_r[0],neg_r[1]]; 
    plt.imshow(r_er_Mtx,origin='lower',cmap='bwr',vmin=-1,vmax=1);   
    plt.colorbar(fraction=0.046, pad=0.04,label='Correlation coefficient (r)')
    plt.title("ShapeResp: r2er"); 

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
            nSig.append(i)
        else:
            plt.plot(x_input, y_pred, color=(0.75,0.75,0.75), linewidth=1);    
    plt.xlabel('Distance between units (micrometer)'); 
    plt.ylabel('Correlation between responses'); 
    plt.title(f"{len(nSig)} / {nNeurons}")    

    plt.subplot(2,3,3); 
    r_Mtx = r2_Mtx**0.5; 
    r_Mtx[np.isnan(r_Mtx)] = 0; 

    r_Mtx[neg_r[0],neg_r[1]] = -1*r_Mtx[neg_r[0],neg_r[1]]; 
    plt.imshow(r_Mtx,origin='lower',cmap='bwr',vmin=-1,vmax=1);   
    plt.colorbar(fraction=0.046, pad=0.04,label='Correlation coefficient (r)')
    plt.title('ShapeResp: r'); 

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
    plt.savefig(path_to_save+"ShapeCluster.pdf", format="pdf", bbox_inches="tight"); 

    name_to_save = path_to_save + bin_filename[(bin_filename.rfind('/')+1):-8] + 'json.gz';
    f = gzip.GzipFile(name_to_save,'w');    
    f.write(json.dumps(experiment, cls=NumpyEncoder).encode('utf-8')); 
    f.close(); 
    print('processed file was saved'); 


    #%% Stimulus conditions: center stim has a texture surface
    """
    shapes_position1 = np.arange(0,250,5); 
    shapes_position2 = np.arange(1,250,5); 
    shapes_position3 = np.arange(2,250,5); 
    shapes_position4 = np.arange(3,250,5); 
    shapes_position5 = np.arange(4,250,5); 

    NoStim = 500; 
    """


#%% Drawing part
    # sort neurons according to response level (strong to weak)
    neurons_from_strong = np.sum(mResp,axis=0).argsort()[::-1]; 

    """
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

        for shapes in np.arange(50):
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
        ax1.imshow(psth_pos1,vmin=0,vmax=yMax,aspect='auto'); 
        ax3.clear(); 
        ax3.imshow(psth_pos2,vmin=0,vmax=yMax,aspect='auto'); 
        ax5.clear(); 
        ax5.imshow(psth_pos3,vmin=0,vmax=yMax,aspect='auto'); 
        ax7.clear(); 
        ax7.imshow(psth_pos4,vmin=0,vmax=yMax,aspect='auto'); 
        ax9.clear(); 
        ax9.imshow(psth_pos5,vmin=0,vmax=yMax,aspect='auto'); 


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
"""
filename = '/Volumes/TK_exHDD3/Anesthetized_V2_Jul2023/P08/P08_M140_combined_03-04-09-10/processed/M140_230718_PosInvariance_RE_NPX_g8_t1.json.gz'; 

f = gzip.GzipFile(filename,'r'); 
Data = json.loads(f.read().decode('utf-8')); 
f.close(); 

mResp = np.array(Data['mResp'])

matrixA = []; 
matrixSig = []; 
for r in range(np.shape(mResp)[1]):
    numerator = 0; 
    denominator = 0; 
    for i in range(4):
        respA = mResp[np.arange(i,250,5),r] - np.mean(mResp[np.arange(i,250,5),r]); 
        for j in range(i+1,5):
            respB = mResp[np.arange(j,250,5),r] - np.mean(mResp[np.arange(j,250,5),r]);      
            numerator += np.cov(respA,respB)[0,1]; 
            denominator += np.std(respA)*np.std(respB); 
    matrixA.append(numerator/denominator); 

    simul = []; 
    for s in range(100):
        numerator = 0; 
        denominator = 0; 
        for i in range(4):
            respA = mResp[np.arange(i,250,5),r] - np.mean(mResp[np.arange(i,250,5),r]); 
            np.random.shuffle(respA); 
            for j in range(i+1,5):
                respB = mResp[np.arange(j,250,5),r] - np.mean(mResp[np.arange(j,250,5),r]);      
                np.random.shuffle(respB); 
                numerator += np.cov(respA,respB)[0,1]; 
                denominator += np.std(respA)*np.std(respB); 
        simul.append(numerator/denominator); 
    simul = np.array(simul); 
    bigger_than_random = len(np.where(simul > matrixA[-1])[0]); 
    if bigger_than_random < 5:
        matrixSig.append(1); 
    else:
        matrixSig.append(0); 

    print(f'neuron: {r} was done');         

matrixA = np.array(matrixA); 
matrixSig = np.array(matrixSig); 

signi = np.where(matrixSig==1)[0]; 
plt.hist(matrixA,np.arange(-1,1.1,0.05))
plt.hist(matrixA[signi],np.arange(-1,1.1,0.05)); 

"""

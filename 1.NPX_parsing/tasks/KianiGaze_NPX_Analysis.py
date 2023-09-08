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
    numStims = 204;     """(50 stims + 1 blank) x 4 fixations """
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
        print(f'condition {i}/208 was done')    

            
    del experiment['stimStructs'];
    del experiment['iti_start'];
    del experiment['iti_end'];
    #del experiment['iti'];
    experiment['filename'] = dat_filename;
    experiment['StimResp'] = StimResp;
    experiment['mResp'] = mResp;    

    pos1_resp = mResp[:51,:]; 
    pos2_resp = mResp[51:102,:]; 
    pos3_resp = mResp[102:153,:]; 
    pos4_resp = mResp[153:,:]; 

    depth = np.array(experiment['chpos_sua'])[:,1]; 
    depth_order = depth.argsort();   # from the tip to the top

    
    ### drawing part
    plt.figure(figsize=(9,10)); 
    ax1 = plt.subplot(3,3,1); 
    ax2 = plt.subplot(3,3,2); 
    ax3 = plt.subplot(3,3,3); 
    ax5 = plt.subplot(3,3,5); 
    ax6 = plt.subplot(3,3,6); 
    ax9 = plt.subplot(3,3,9); 

    ax4 = plt.subplot(3,3,4);   # std comparison     
    ax7 = plt.subplot(3,3,7);   # object preference      

    neurons_from_strong = np.sum(mResp,axis=0).argsort()[::-1]; 
    nClusters = experiment['numNeurons']; 
    for jj in np.arange(experiment['numNeurons']):
    
        plt.gcf().suptitle('This is a somewhat long figure title', fontsize=16)

        j = neurons_from_strong[jj]; 
        if app.all_radiobutton.isChecked() == True:
            unit_id = experiment['neuronid'][j]; 
        elif app.sua_radiobutton.isChecked() == True:
            unit_id = experiment['id_sua'][j]; 
        elif app.mua_radiobutton.isChecked() == True:
            unit_id = experiment['id_mua'][j]; 

        draw_figure(ax1, 1, 2, j, mResp); 
        draw_figure(ax2, 1, 3, j, mResp);         
        draw_figure(ax3, 1, 4, j, mResp);                 
        draw_figure(ax5, 2, 3, j, mResp);                         
        draw_figure(ax6, 2, 4, j, mResp);                         
        draw_figure(ax9, 3, 4, j, mResp); 

        ax4.clear(); 
        ax4.bar([1,2,3,4],[np.std(pos1_resp[:,j]),np.std(pos2_resp[:,j]),
                           np.std(pos3_resp[:,j]),np.std(pos4_resp[:,j])]); 
        ax4.set_xlabel('Gaze position')
        ax4.set_ylabel('Std of responses')

        ax7.clear()
        stim_rank = pos1_resp[:,j].argsort()[::-1]; 
        ax7.plot(pos1_resp[stim_rank,j], 'C0', label='pos1')
        ax7.plot(pos2_resp[stim_rank,j], 'C1', label='pos2')
        ax7.plot(pos3_resp[stim_rank,j], 'C2', label='pos3')
        ax7.plot(pos4_resp[stim_rank,j], 'C3', label='pos4')                        
        ax7.legend(); 
        ax7.set_xlabel('Stimulus preference')
        ax7.set_ylabel('Responses')        
        Fval, pval = stats.f_oneway(pos1_resp[:,j], pos2_resp[:,j],
                                    pos3_resp[:,j], pos4_resp[:,j]); 
        ax7.set_title(f'one-way ANOVA. p = {round(pval,2)}')

        plt.tight_layout();       
        plt.pause(1); 

        if app.running == 0:
            break; 
    plt.show(block=True);

    ### ANOVA
    f,pval = stats.f_oneway(pos1_resp,pos2_resp,pos3_resp,pos4_resp);     


    """
    f,pval = stats.f_oneway(pos1_resp,pos2_resp,pos3_resp,pos4_resp);     

    sigCells = np.where(pval<0.05)[0]; 

    for i in range(len(sigCells)):
        unitID = sigCells[i]; 

        stim_rank = pos1_resp[:,unitID].argsort()[::-1];      
        yMax = int(np.max(np.array(Data['mResp'])[:,unitID])); 

        r12, pval12 = stats.pearsonr(pos1_resp[:,unitID],pos2_resp[:,unitID]); 
        r13, pval13 = stats.pearsonr(pos1_resp[:,unitID],pos3_resp[:,unitID]); 
        r14, pval14 = stats.pearsonr(pos1_resp[:,unitID],pos4_resp[:,unitID]);     

        if (yMax>10) and (np.max([pval12,pval13,pval14])<0.05):
            plt.figure(figsize=(9,6))
            plt.subplot(2,1,1)
            plt.plot(pos1_resp[stim_rank,unitID],label='pos1')
            plt.plot(pos2_resp[stim_rank,unitID],label='pos2')
            plt.plot(pos3_resp[stim_rank,unitID],label='pos3')
            plt.plot(pos4_resp[stim_rank,unitID],label='pos4')
            plt.legend()  
            plt.xlabel('Stim preference'); 
            plt.ylabel('Responses (Hz)'); 
            plt.title(f'unitID = {unitID}')
            plt.gca().spines[['right', 'top']].set_visible(False)

            plt.subplot(2,3,4)
            plt.plot(pos1_resp[stim_rank,unitID],pos2_resp[stim_rank,unitID],'k.')
            plt.plot([0,yMax],[0,yMax],'r')
            plt.xlabel('Pos1 responses'); 
            plt.ylabel('Pos2 responses'); 
            plt.gca().spines[['right', 'top']].set_visible(False)
            plt.title(f'r = {round(r12,2)}. p = {round(pval12,2)}')
            plt.subplot(2,3,5)
            plt.plot(pos1_resp[stim_rank,unitID],pos3_resp[stim_rank,unitID],'k.')
            plt.plot([0,yMax],[0,yMax],'r')
            plt.xlabel('Pos1 responses'); 
            plt.ylabel('Pos3 responses'); 
            plt.gca().spines[['right', 'top']].set_visible(False)
            plt.title(f'r = {round(r13,2)}. p = {round(pval13,2)}')        
            plt.subplot(2,3,6)
            plt.plot(pos1_resp[stim_rank,unitID],pos4_resp[stim_rank,unitID],'k.')
            plt.plot([0,yMax],[0,yMax],'r')
            plt.xlabel('Pos1 responses'); 
            plt.ylabel('Pos4 responses'); 
            plt.gca().spines[['right', 'top']].set_visible(False)
            plt.title(f'r = {round(r13,2)}. p = {round(pval13,2)}')        
            plt.tight_layout()
    """


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
       
#%%
"""
import numpy as np;
import json;
import gzip;
import matplotlib.pyplot as plt; 

#from sklearn import linear_model
from scipy import stats
import statsmodels.api as sm

#filename = '/Volumes/TK_exHDD1/NPX/Kiani_Gaze/x230629_combined/processed/x230629_KianiObject_g0_t0.json.gz'; 
filename = '/Volumes/TK_exHDD1/NPX/Kiani_Gaze/x230630_combined/processed/x230630_KianiGaze_g1_t1.json.gz';
#filename = '/Volumes/TK_exHDD1/NPX/Kiani_Gaze/x230703_combined/processed/x230703_KianiObject_g1_t1.json.gz'; 
#filename = '/Volumes/TK_exHDD1/NPX/Kiani_Gaze/x230711_combined/processed/x230711_KianiObject_g0_t0.json.gz'; 
f = gzip.GzipFile(filename,'r');
Data = json.loads(f.read().decode('utf-8'));
f.close();

mResp = np.array(Data['mResp'])
pos1_resp = mResp[:51,:]; 
pos2_resp = mResp[51:102,:]; 
pos3_resp = mResp[102:153,:]; 
pos4_resp = mResp[153:,:]; 

mResp = np.array(Data['mResp'])
pos1_resp = mResp[:51,:]; 
pos2_resp = mResp[51:102,:]; 
pos3_resp = mResp[102:153,:]; 
pos4_resp = mResp[153:,:]; 

f,pval = stats.f_oneway(pos1_resp,pos2_resp,pos3_resp,pos4_resp);     

sigCells = np.where(pval<0.05)[0]; 
for i in range(len(sigCells)):
    unitID = sigCells[i]; 

    stim_rank = pos1_resp[:,unitID].argsort()[::-1];      
    yMax = int(np.max(np.array(Data['mResp'])[:,unitID])); 

    r12, pval12 = stats.pearsonr(pos1_resp[:,unitID],pos2_resp[:,unitID]); 
    r13, pval13 = stats.pearsonr(pos1_resp[:,unitID],pos3_resp[:,unitID]); 
    r14, pval14 = stats.pearsonr(pos1_resp[:,unitID],pos4_resp[:,unitID]);     

    meanSDF = np.empty((0,900)); 
    for j in range(204):
        meanSDF = np.vstack((meanSDF,np.array(Data['StimResp'][j]['neurons'][unitID]['meanSDF']).reshape((1,900)))); 
    yMax1 = np.max(np.mean(meanSDF,axis=0));


    if (yMax1>5): # and (np.max([pval12,pval13,pval14])<0.05):
        plt.figure(figsize=(9,6))
        plt.subplot(2,1,1)
        plt.plot(pos1_resp[stim_rank,unitID],label='pos1')
        plt.plot(pos2_resp[stim_rank,unitID],label='pos2')
        plt.plot(pos3_resp[stim_rank,unitID],label='pos3')
        plt.plot(pos4_resp[stim_rank,unitID],label='pos4')
        plt.legend()  
        plt.xlabel('Stim preference'); 
        plt.ylabel('Responses (Hz)'); 
        plt.title(f'unitID = {unitID}')
        plt.gca().spines[['right', 'top']].set_visible(False)

        plt.subplot(2,3,4)
        plt.plot(pos1_resp[stim_rank,unitID],pos2_resp[stim_rank,unitID],'k.')
        plt.plot([0,yMax],[0,yMax],'r')
        plt.xlabel('Pos1 responses'); 
        plt.ylabel('Pos2 responses'); 
        plt.gca().spines[['right', 'top']].set_visible(False)
        plt.title(f'r = {round(r12,2)}. p = {round(pval12,2)}')
        plt.subplot(2,3,5)
        plt.plot(pos1_resp[stim_rank,unitID],pos3_resp[stim_rank,unitID],'k.')
        plt.plot([0,yMax],[0,yMax],'r')
        plt.xlabel('Pos1 responses'); 
        plt.ylabel('Pos3 responses'); 
        plt.gca().spines[['right', 'top']].set_visible(False)
        plt.title(f'r = {round(r13,2)}. p = {round(pval13,2)}')        
        plt.subplot(2,3,6)
        plt.plot(pos1_resp[stim_rank,unitID],pos4_resp[stim_rank,unitID],'k.')
        plt.plot([0,yMax],[0,yMax],'r')
        plt.xlabel('Pos1 responses'); 
        plt.ylabel('Pos4 responses'); 
        plt.gca().spines[['right', 'top']].set_visible(False)
        plt.title(f'r = {round(r14,2)}. p = {round(pval14,2)}')        
        plt.tight_layout()


        plt.figure(figsize=(9,3)); 
        gStim = stim_rank[:25];
        bStim = stim_rank[25:];  
    
        yMax2 = np.max(np.mean(meanSDF[gStim,:],axis=0)); 
        for j in range(4):
            plt.subplot(1,4,j+1); 
            plt.plot(np.mean(meanSDF[gStim+j*51,:],axis=0),'r'); 
            plt.plot(np.mean(meanSDF[bStim+j*51,:],axis=0),'b'); 
            plt.plot(np.mean(meanSDF[j*51:(j+1)*51,:],axis=0),'k');             
            plt.ylim([0,yMax2]); 
            plt.title(f'Position {j+1}')
        plt.tight_layout(); 
        print(f'unitID: {unitID} was done')       
"""
"""
import numpy as np;
import json;
import gzip;
import matplotlib.pyplot as plt; 

#from sklearn import linear_model
from scipy import stats
import statsmodels.api as sm

#filename = '/Volumes/TK_exHDD1/NPX/Kiani_Gaze/x230629_combined/processed/x230629_KianiObject_g0_t0.json.gz'; 
filename = '/Volumes/TK_exHDD1/NPX/Kiani_Gaze/x230630_combined/processed/x230630_KianiGaze_g1_t1.json.gz';
#filename = '/Volumes/TK_exHDD1/NPX/Kiani_Gaze/x230703_combined/processed/x230703_KianiObject_g1_t1.json.gz'; 
#filename = '/Volumes/TK_exHDD1/NPX/Kiani_Gaze/x230711_combined/processed/x230711_KianiObject_g0_t0.json.gz'; 
f = gzip.GzipFile(filename,'r');
Data = json.loads(f.read().decode('utf-8'));
f.close();

pos = [(0,0),(5,0),(5,5),(0,5)]; 

mResp = np.array(Data['mResp']); 
pos1_resp = mResp[:51,:]; 
pos2_resp = mResp[51:102,:]; 
pos3_resp = mResp[102:153,:]; 
pos4_resp = mResp[153:,:]; 
posAvg_resp = (pos1_resp + pos2_resp + pos3_resp + pos4_resp)/4; 

f,pval = stats.f_oneway(pos1_resp,pos2_resp,pos3_resp,pos4_resp);     

sigCells = np.where(pval<0.05)[0]; 
for i in range(len(sigCells)):
    unitID = sigCells[i]; 

    meanSDF = np.empty((0,900)); 
    for j in range(204):
        meanSDF = np.vstack((meanSDF,np.array(Data['StimResp'][j]['neurons'][unitID]['meanSDF']).reshape((1,900)))); 

    stim_rank = pos1_resp[:,unitID].argsort()[::-1];      
    gStim = stim_rank[:25];
    bStim = stim_rank[25:];  

    yMax = np.max(mResp[:,unitID]); 
    yMax2 = np.max([np.mean(meanSDF[gStim,:],axis=0),
                    np.mean(meanSDF[gStim+51,:],axis=0),
                    np.mean(meanSDF[gStim+51*2,:],axis=0),
                    np.mean(meanSDF[gStim+51*3,:],axis=0)]); 

    if yMax2 < 10:
        continue; 

    plt.figure(figsize=(12,12)); 
    if Data['neuronid'][unitID] in Data['id_sua']:
        plt.gcf().suptitle(f"unitID = {Data['neuronid'][unitID]}. SUA", fontsize=16)
    elif Data['neuronid'][unitID] in Data['id_mua']:    
        plt.gcf().suptitle(f"unitID = {Data['neuronid'][unitID]}. MUA", fontsize=16)    
    plt.subplot(4,4,6); 
    plt.plot(np.mean(meanSDF[gStim,:],axis=0),'C0',ls='-'); 
    plt.plot(np.mean(meanSDF[bStim,:],axis=0),'C0',ls=':'); 
    plt.ylim([0, yMax2]);
    plt.xticks(np.arange(0,1200,300),np.arange(-300,900,300)); 
    plt.xlabel('Time from stimulus onset (ms)'); 
    plt.title(f"Gaze position1: {pos[0]}"); 
    plt.gca().spines[['right', 'top']].set_visible(False)

    plt.subplot(4,4,7); 
    plt.plot(mResp[:51,unitID],mResp[51:51*2,unitID],'k.'); 
    plt.plot([0, yMax],[0, yMax],'r');
    plt.xlabel('Response at Pos 1'); 
    plt.ylabel('Response at Pos 2'); 
    plt.title(f"Gaze position2: {pos[1]}"); 
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.subplot(4,4,8); 
    plt.plot(np.mean(meanSDF[gStim+51,:],axis=0),'C1',ls='-'); 
    plt.plot(np.mean(meanSDF[bStim+51,:],axis=0),'C1',ls=':'); 
    plt.ylim([0, yMax2]);
    plt.xticks(np.arange(0,1200,300),np.arange(-300,900,300)); 
    plt.xlabel('Time from stimulus onset (ms)'); 
    plt.title(f"Gaze position2: {pos[1]}"); 
    plt.gca().spines[['right', 'top']].set_visible(False)

    plt.subplot(4,4,3); 
    plt.plot(mResp[:51,unitID],mResp[51*2:51*3,unitID],'k.'); 
    plt.plot([0, yMax],[0, yMax],'r');
    plt.xlabel('Response at Pos 1'); 
    plt.ylabel('Response at Pos 3'); 
    plt.title(f"Gaze position3: {pos[2]}"); 
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.subplot(4,4,4); 
    plt.plot(np.mean(meanSDF[gStim+51*2,:],axis=0),'C2',ls='-'); 
    plt.plot(np.mean(meanSDF[bStim+51*2,:],axis=0),'C2',ls=':'); 
    plt.ylim([0, yMax2]);
    plt.xticks(np.arange(0,1200,300),np.arange(-300,900,300)); 
    plt.xlabel('Time from stimulus onset (ms)'); 
    plt.title(f"Gaze position3: {pos[2]}"); 
    plt.gca().spines[['right', 'top']].set_visible(False)

    plt.subplot(4,4,1); 
    plt.plot(mResp[:51,unitID],mResp[51*3:51*4,unitID],'k.'); 
    plt.plot([0, yMax],[0, yMax],'r');
    plt.xlabel('Response at Pos 1'); 
    plt.ylabel('Response at Pos 4'); 
    plt.title(f"Gaze position4: {pos[3]}"); 
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.subplot(4,4,2); 
    plt.plot(np.mean(meanSDF[gStim+51*3,:],axis=0),'C3',ls='-'); 
    plt.plot(np.mean(meanSDF[bStim+51*3,:],axis=0),'C3',ls=':'); 
    plt.ylim([0, yMax2]);
    plt.xticks(np.arange(0,1200,300),np.arange(-300,900,300)); 
    plt.xlabel('Time from stimulus onset (ms)'); 
    plt.title(f"Gaze position4: {pos[3]}"); 
    plt.gca().spines[['right', 'top']].set_visible(False)

    plt.subplot(4,4,(9,10)); 
    plt.imshow(meanSDF[stim_rank+51*3,:],vmin=0,vmax=np.max(meanSDF)*0.7,aspect='auto'); 
    plt.xlabel('Time (ms)'); 
    plt.ylabel('Stim Rank'); 
    plt.xticks(np.arange(0,1200,300),np.arange(-300,900,300)); 
    plt.xlabel('Time from stimulus onset (ms)'); 
    plt.title(f"Gaze position4: {pos[3]}")

    plt.subplot(4,4,(11,12)); 
    plt.imshow(meanSDF[stim_rank+51*2,:],vmin=0,vmax=np.max(meanSDF)*0.7,aspect='auto'); 
    plt.xlabel('Time (ms)'); 
    plt.ylabel('Stim Rank'); 
    plt.xticks(np.arange(0,1200,300),np.arange(-300,900,300)); 
    plt.xlabel('Time from stimulus onset (ms)'); 
    plt.title(f"Gaze position3: {pos[2]}")

    plt.subplot(4,4,(13,14)); 
    plt.imshow(meanSDF[stim_rank+51*0,:],vmin=0,vmax=np.max(meanSDF)*0.7,aspect='auto'); 
    plt.xlabel('Time (ms)'); 
    plt.ylabel('Stim Rank'); 
    plt.xticks(np.arange(0,1200,300),np.arange(-300,900,300)); 
    plt.xlabel('Time from stimulus onset (ms)'); 
    plt.title(f"Gaze position1: {pos[0]}")

    plt.subplot(4,4,(15,16)); 
    plt.imshow(meanSDF[stim_rank+51*1,:],vmin=0,vmax=np.max(meanSDF)*0.7,aspect='auto'); 
    plt.xlabel('Time (ms)'); 
    plt.ylabel('Stim Rank'); 
    plt.xticks(np.arange(0,1200,300),np.arange(-300,900,300)); 
    plt.xlabel('Time from stimulus onset (ms)'); 
    plt.title(f"Gaze position2: {pos[1]}")
    plt.tight_layout()

"""
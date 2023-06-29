#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:38:14 2019

ClutterStim_NPX_Analysis.py

@author: taekjunkim
"""

import matplotlib.pyplot as plt;
import scipy.optimize as opt;

#import sys;
#sys.path.append('./helper'); 

import numpy as np;
import makeSDF;
import glob; 
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

    task_index_in_combine = int(app.tasknum_lineEdit.text()); 

    #%% get experiment
    prevTime = 0.3; 
    numStims = 1380; 
    experiment = parse_NPX.main(bin_filename, dat_filename, prevTime, numStims, imec_filename, app); 
    experiment['StimDur'] = experiment['stimon']; 

    TimeOfInterest = np.arange(int(prevTime*1000+50),int(prevTime*1000+experiment['StimDur']+100+1));
    
    StimResp = [];
    mResp = np.zeros((numStims,experiment['numNeurons']));
    for i in np.arange(len(experiment['stimStructs'])):
        CondNum = int(i%138);
        RepNum = int(np.floor(i/138))+1;
        
        if RepNum==1:
            StimResp.append(dict());
            StimResp[CondNum]['timeOn'] = experiment['stimStructs'][i]['timeOn'];
            StimResp[CondNum]['timeOff'] = experiment['stimStructs'][i]['timeOff'];    
            StimResp[CondNum]['pdOn'] = experiment['stimStructs'][i]['pdOn'];
            StimResp[CondNum]['pdOff'] = experiment['stimStructs'][i]['pdOff'];     
            StimResp[CondNum]['neurons'] = experiment['stimStructs'][i]['neurons'];    
            if experiment['stimStructs'][i]['numInstances']==0:
                for j in np.arange(experiment['numNeurons']):
                    StimResp[CondNum]['neurons'][j]['spikes'] = [];
        elif experiment['stimStructs'][i]['numInstances']>0:
            StimResp[CondNum]['timeOn'].append(experiment['stimStructs'][i]['timeOn'][0]);
            StimResp[CondNum]['timeOff'].append(experiment['stimStructs'][i]['timeOff'][0]);    
            StimResp[CondNum]['pdOn'].append(experiment['stimStructs'][i]['pdOn'][0]);
            StimResp[CondNum]['pdOff'].append(experiment['stimStructs'][i]['pdOff'][0]);     
            for j in np.arange(experiment['numNeurons']):
                StimResp[CondNum]['neurons'][j]['spikes'].append(experiment['stimStructs'][i]['neurons'][j]['spikes'][0]);
    print('StimResp was made');                 
            
    for i in np.arange(len(StimResp)):
        for j in np.arange(experiment['numNeurons']):
        #for j in np.arange(1):        
            NumRepeat = len(StimResp[i]['pdOn']);
            sigLength = int(experiment['StimDur'] + experiment['prevTime']*1000 
                        + experiment['postTime']*1000);       
            StimResp[i]['neurons'][j]['spkMtx'] = np.zeros((NumRepeat,sigLength),dtype=int);
            StimResp[i]['neurons'][j]['numspikes'] = np.zeros((NumRepeat,1),dtype=int);        
            for r in np.arange(NumRepeat):
                spkTime = StimResp[i]['neurons'][j]['spikes'][r] - StimResp[i]['pdOn'][r];
                spkTime = spkTime[:]*1000 + experiment['prevTime']*1000;
                spkTime = spkTime[np.where(spkTime[:]<sigLength)];
                spkTime = spkTime.astype(int);
                StimResp[i]['neurons'][j]['spkMtx'][r,spkTime] = 1;
                StimResp[i]['neurons'][j]['numspikes'][r] = np.sum(StimResp[i]['neurons'][j]['spkMtx'][r,TimeOfInterest]);            
            
            StimResp[i]['neurons'][j]['meanSDF'] = makeSDF.getSDF(StimResp[i]['neurons'][j]['spkMtx'],1000);
            mResp[i,j] = np.mean(StimResp[i]['neurons'][j]['meanSDF'][TimeOfInterest])
    print('meanSDF, mResp were computed');                 

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

    #%% Stimulus conditions
    """
    No stim (0)
    Gray Center Alone (1-8)
    Gray Surround 1 Near Alone (9)
    Gray Center + Gray Surround 1 Near (10-17)
    Gray Surround 3 Near Alone (18)
    Gray Center + Gray Surround 3 Near (19-26)
    Gray Surround 6 Near Alone (27)
    Gray Center + Gray Surround 6 Near (28-35)
    Gray Surround 12 Middle Alone (36)
    Gray Center + Gray Surround 12 Middle (37-44)
    Gray Surround 18 Far Alone (45)
    Gray Center + Gray Surround 18 Far (46-53)
    Gray Surround 6 Circle Near Alone (54)
    Gray Center + Gray Surround 6 Circle Near (55-62)
    Gray Surround 12 Small Near Alone (63)
    Gray Center + Gray Surround 12 Small Near (64-71)
    Gray Surround 12 Small Circle Near Alone (72)
    Gray Center + Gray Surround 12 Small Circle Near (73-80)
    Color Center Alone (81-88)
    Color Center + Gray Surround 6 Near (89-96)
    Color Surround 6 Near Alone (97)
    Gray Center + Color Surround 6 Near (98-105)
    Gray Center + Gray Surround 6 Near (PS) (106-113)
    Gray Center + Gray Surround 6 Circle Near (PS) (114-121)
    Gray Center + Gray Surround 12 Small Near (PS) (122-129)
    Gray Center + Gray Surround 12 Small Circle Near (PS) (130-137)
    """

#%% Drawing part
    plt.figure(figsize=(12,9)); 
    ax1 = plt.subplot(3,4,1); 
    ax2 = plt.subplot(3,4,2); 
    ax3 = plt.subplot(3,4,3); 
    ax4 = plt.subplot(3,4,4); 
    ax5 = plt.subplot(3,4,5);     
    ax6 = plt.subplot(3,4,6);         
    ax7 = plt.subplot(3,4,7);     
    ax8 = plt.subplot(3,4,8);            
    ax9 = plt.subplot(3,4,9);     
    ax10 = plt.subplot(3,4,10);         
    ax11 = plt.subplot(3,4,11);     
    ax12 = plt.subplot(3,4,12);            

    # sort neurons according to response level (strong to weak)
    neurons_from_strong = np.sum(mResp,axis=0).argsort()[::-1]; 
    nClusters = experiment['numNeurons']; 

    # strong 5 neurons
    for jj in np.arange(nClusters):

        j = neurons_from_strong[jj]; 
        if app.all_radiobutton.isChecked() == True:
            unit_id = experiment['neuronid'][j]; 
        elif app.sua_radiobutton.isChecked() == True:
            unit_id = experiment['id_sua'][j]; 
        elif app.mua_radiobutton.isChecked() == True:
            unit_id = experiment['id_mua'][j]; 
        
        NoStim = np.mean(StimResp[0]['neurons'][j]['meanSDF'][TimeOfInterest]);
        
        ## Control of near surround stimulus number
        gC = mResp[1:9,j];
        gC_gS1N = mResp[10:18,j];
        gC_gS3N = mResp[19:27,j];
        gC_gS6N = mResp[28:36,j];

        gC_ste = [];
        gC_gS1N_ste = [];   
        gC_gS3N_ste = [];
        gC_gS6N_ste = [];
        for i in np.arange(8):
            gC_ste.append((np.std(StimResp[1+i]['neurons'][j]['numspikes'])*1000/350)
                                /(np.sqrt(len(StimResp[1+i]['neurons'][j]['numspikes']))));
            gC_gS1N_ste.append((np.std(StimResp[10+i]['neurons'][j]['numspikes'])*1000/350)
                                /(np.sqrt(len(StimResp[10+i]['neurons'][j]['numspikes']))));
            gC_gS3N_ste.append((np.std(StimResp[19+i]['neurons'][j]['numspikes'])*1000/350)
                                /(np.sqrt(len(StimResp[19+i]['neurons'][j]['numspikes']))));
            gC_gS6N_ste.append((np.std(StimResp[28+i]['neurons'][j]['numspikes'])*1000/350)
                                /(np.sqrt(len(StimResp[28+i]['neurons'][j]['numspikes']))));
        rk = np.flip(np.argsort(gC)); ## ranking of shape preference    
        
        ### # of Surround (3,4,1)
        ax1.clear()
        ax1.errorbar(np.arange(1,9),gC[rk],np.array(gC_ste)[rk],color=[1,0,0]); 
        ax1.errorbar(np.arange(1,9),gC_gS1N[rk],np.array(gC_gS1N_ste)[rk],color=[0.6,0.6,0.6]); 
        ax1.errorbar(np.arange(1,9),gC_gS3N[rk],np.array(gC_gS3N_ste)[rk],color=[0.3,0.3,0.3]); 
        ax1.errorbar(np.arange(1,9),gC_gS6N[rk],np.array(gC_gS6N_ste)[rk],color=[0,0,0]);    
        ax1.set_xlabel('Stim ID')
        ax1.set_ylabel('Responses (spk/s)')

        if unit_id in experiment['id_sua']:
            ax1.set_title(f'unit#{unit_id} (SUA): # of N.Surr'); 
        elif unit_id in experiment['id_mua']:    
            ax1.set_title(f'unit#{unit_id} (MUA): # of N.Surr'); 
        
        ### # of Surround: PSTH (3,4,5)
        gC_sdf = np.zeros((900,));
        gC_gS1N_sdf = np.zeros((900,));
        gC_gS3N_sdf = np.zeros((900,));
        gC_gS6N_sdf = np.zeros((900,));    
        for i in np.arange(8):
            gC_sdf += StimResp[1+i]['neurons'][j]['meanSDF'];
            gC_gS1N_sdf += StimResp[10+i]['neurons'][j]['meanSDF'];
            gC_gS3N_sdf += StimResp[19+i]['neurons'][j]['meanSDF'];
            gC_gS6N_sdf += StimResp[28+i]['neurons'][j]['meanSDF'];      
        ax5.clear()              
        ax5.plot(np.arange(-100,500),gC_sdf[200:800]/8,color=[1,0,0]);
        ax5.plot(np.arange(-100,500),gC_gS1N_sdf[200:800]/8,color=[0.6,0.6,0.6]);
        ax5.plot(np.arange(-100,500),gC_gS3N_sdf[200:800]/8,color=[0.3,0.3,0.3]);
        ax5.plot(np.arange(-100,500),gC_gS6N_sdf[200:800]/8,color=[0,0,0]);        
        ax5.set_xlabel('Time from stimulus onset (ms)')
        ax5.set_ylabel('Response (Hz)')        

        ### # of Surround: Modulation (3,4,9)
        modMtx1 = np.empty((3,900));
        modMtx1[0,:] = gC_gS1N_sdf[:];
        modMtx1[1,:] = gC_gS3N_sdf[:];
        modMtx1[2,:] = gC_gS6N_sdf[:];
        modSD1 = np.std(modMtx1,axis=0);
        ax9.clear()
        ax9.plot(np.arange(-100,500),modSD1[200:800]/8,color=[0,0,1]);
        ax9.set_xlabel('Time from stimulus onset (ms)')
        ax9.set_ylabel('Modulation (std)')        
        del modMtx1, modSD1;

        ## center-surround distance (3,4,2)
        gC_gS12M = mResp[37:45,j];
        gC_gS18F = mResp[46:54,j];    
        gC_gS12M_ste = [];   gC_gS18F_ste = [];
        for i in np.arange(8):
            gC_gS12M_ste.append((np.std(StimResp[37+i]['neurons'][j]['numspikes'])*1000/350)
                                /(np.sqrt(len(StimResp[37+i]['neurons'][j]['numspikes']))));
            gC_gS18F_ste.append((np.std(StimResp[46+i]['neurons'][j]['numspikes'])*1000/350)
                                /(np.sqrt(len(StimResp[46+i]['neurons'][j]['numspikes']))));

        ax2.clear()     
        ax2.errorbar(np.arange(1,9),gC[rk],np.array(gC_ste)[rk],color=[1,0,0]);
        ax2.errorbar(np.arange(1,9),gC_gS6N[rk],np.array(gC_gS6N_ste)[rk],color=[0,0,0]);
        ax2.errorbar(np.arange(1,9),gC_gS12M[rk],np.array(gC_gS12M_ste)[rk],color=[0.3,0.3,0.3]);
        ax2.errorbar(np.arange(1,9),gC_gS18F[rk],np.array(gC_gS18F_ste)[rk],color=[0.6,0.6,0.6]);    
        ax2.set_title('Distance');    
        
        # center-surround distance: PSTH (3,4,6)
        gC_gS12M_sdf = np.zeros((900,));
        gC_gS18F_sdf = np.zeros((900,));
        for i in np.arange(8):
            gC_gS12M_sdf += StimResp[37+i]['neurons'][j]['meanSDF'];
            gC_gS18F_sdf += StimResp[46+i]['neurons'][j]['meanSDF'];        
        ax6.clear()
        ax6.plot(np.arange(-100,500),gC_sdf[200:800]/8,color=[1,0,0]);
        ax6.plot(np.arange(-100,500),gC_gS18F_sdf[200:800]/8,color=[0.6,0.6,0.6]);
        ax6.plot(np.arange(-100,500),gC_gS12M_sdf[200:800]/8,color=[0.3,0.3,0.3]);
        ax6.plot(np.arange(-100,500),gC_gS6N_sdf[200:800]/8,color=[0,0,0]);       
        ax6.set_xlabel('Time from stimulus onset (ms)')
        ax6.set_ylabel('Response (Hz)')        
        
        # center-surround distance: Modulation (3,4,10)
        modMtx2 = np.empty((3,900));
        modMtx2[0,:] = gC_gS6N_sdf[:];
        modMtx2[1,:] = gC_gS12M_sdf[:];
        modMtx2[2,:] = gC_gS18F_sdf[:];
        modSD2 = np.std(modMtx2,axis=0);
        ax10.clear()
        ax10.plot(np.arange(-100,500),modSD2[200:800]/8,color=[0,0,1]); 
        ax10.set_xlabel('Time from stimulus onset (ms)')
        ax10.set_ylabel('Modulation (std)')        
        del modMtx2, modSD2;

        ## Saliency control by surround (3,4,3)
        gC_gS6CN = mResp[55:63,j];
        gC_gS12SN = mResp[64:72,j];    
        gC_gS12SCN = mResp[73:81,j];        
        
        gC_gS6CN_ste = [];
        gC_gS12SN_ste = [];    
        gC_gS12SCN_ste = [];        
        for i in np.arange(8):
            gC_gS6CN_ste.append((np.std(StimResp[55+i]['neurons'][j]['numspikes'])*1000/350)
                                /(np.sqrt(len(StimResp[55+i]['neurons'][j]['numspikes']))));
            gC_gS12SN_ste.append((np.std(StimResp[64+i]['neurons'][j]['numspikes'])*1000/350)
                                /(np.sqrt(len(StimResp[64+i]['neurons'][j]['numspikes']))));
            gC_gS12SCN_ste.append((np.std(StimResp[73+i]['neurons'][j]['numspikes'])*1000/350)
                                /(np.sqrt(len(StimResp[73+i]['neurons'][j]['numspikes']))));

        ax3.clear(); 
        ax3.errorbar(np.arange(1,9),gC[rk],np.array(gC_ste)[rk],color=[1,0,0]);
        ax3.errorbar(np.arange(1,9),gC_gS6N[rk],np.array(gC_gS6N_ste)[rk],color=[0,0,0]);
        ax3.errorbar(np.arange(1,9),gC_gS6CN[rk],np.array(gC_gS6CN_ste)[rk],color=[0.2,0.2,0.2]);    
        ax3.errorbar(np.arange(1,9),gC_gS12SN[rk],np.array(gC_gS12SN_ste)[rk],color=[0.4,0.4,0.4]);
        ax3.errorbar(np.arange(1,9),gC_gS12SCN[rk],np.array(gC_gS12SCN_ste)[rk],color=[0.6,0.6,0.6]);        
        ax3.set_title('Size')

        ## Saliency control by surround color (3,4,4)
        cC = mResp[81:89,j];
        cC_gS6N = mResp[89:97,j];    
        gC_cS6N = mResp[98:106,j]
        
        cC_ste = [];
        cC_gS6N_ste = [];
        gC_cS6N_ste = [];    
        for i in np.arange(8):
            cC_ste.append((np.std(StimResp[81+i]['neurons'][j]['numspikes'])*1000/350)
                                /(np.sqrt(len(StimResp[81+i]['neurons'][j]['numspikes']))));
            cC_gS6N_ste.append((np.std(StimResp[89+i]['neurons'][j]['numspikes'])*1000/350)
                                /(np.sqrt(len(StimResp[89+i]['neurons'][j]['numspikes']))));
            gC_cS6N_ste.append((np.std(StimResp[98+i]['neurons'][j]['numspikes'])*1000/350)
                                /(np.sqrt(len(StimResp[98+i]['neurons'][j]['numspikes']))));

        ax4.clear()
        ax4.errorbar(np.arange(1,9),gC[rk],np.array(gC_ste)[rk],color=[1,0,0]);
        ax4.errorbar(np.arange(1,9),gC_gS6N[rk],np.array(gC_gS6N_ste)[rk],color=[0,0,0]);
        ax4.errorbar(np.arange(1,9),gC_cS6N[rk],np.array(gC_cS6N_ste)[rk],color=[0.5,0.5,0.5]); ## gray center + color surround
        ax4.set_title('Surround color')

        ## Saliency control by surround color2 (3,4,8)
        ax8.clear()
        ax8.errorbar(np.arange(1,9),cC[rk],np.array(cC_ste)[rk],color=[0,0,1]);
        ax8.errorbar(np.arange(1,9),cC_gS6N[rk],np.array(cC_gS6N_ste)[rk],color=[0.5,0.5,1]); ## color center + gray surround    
        ax8.set_title('Center color')

        plt.tight_layout()
        plt.pause(1); 

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


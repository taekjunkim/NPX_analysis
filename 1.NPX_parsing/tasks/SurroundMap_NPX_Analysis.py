#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:38:14 2019

SurroundMap_NPX_Analysis.py

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
import os; 

import parseTJexperiment_NPX as parse_NPX;
import json;
import gzip;

def main(app):

    ### This needs to be defined
    dat_filename = app.dat_file; 
    imec_filename = app.imec_file; 
    task_folder = dat_filename[:(dat_filename.rfind('/')+1)]; 
    bin_filename = glob.glob(task_folder+'*.nidq.bin')[0]; 

    task_index_in_combine = int(app.tasknum_lineEdit.text()); 

    ### get experiment
    prevTime = 0.3; 
    numStims = 148; 
    experiment = parse_NPX.main(bin_filename, dat_filename, prevTime, numStims, imec_filename, app); 
    experiment['StimDur'] = experiment['stimon']; 

    TimeOfInterest = np.arange(int(prevTime*1000+50),int(prevTime*1000+experiment['StimDur']+100+1));

    ###
    StimResp = [];
    mResp = np.zeros((numStims,experiment['numNeurons']));
    for i in np.arange(len(experiment['stimStructs'])):
        StimResp.append(dict());
        StimResp[i]['timeOn'] = experiment['stimStructs'][i]['timeOn'];
        StimResp[i]['timeOff'] = experiment['stimStructs'][i]['timeOff'];    
        StimResp[i]['pdOn'] = experiment['stimStructs'][i]['pdOn'];
        StimResp[i]['pdOff'] = experiment['stimStructs'][i]['pdOff'];        
        StimResp[i]['neurons'] = experiment['stimStructs'][i]['neurons'];        

        for j in np.arange(experiment['numNeurons']):
            NumRepeat = len(StimResp[i]['pdOn']);
            sigLength = int(experiment['StimDur'] + experiment['prevTime']*1000 
                        + experiment['postTime']*1000);       
            StimResp[i]['neurons'][j]['spkMtx'] = np.zeros((NumRepeat,sigLength),dtype=int);
            StimResp[i]['neurons'][j]['numspikes'] = np.zeros((NumRepeat,1),dtype=int);                
            for r in np.arange(NumRepeat):
                spkTime = StimResp[i]['neurons'][j]['spikes'][r] - StimResp[i]['pdOn'][r];
                spkTime = spkTime[:]*1000 + experiment['prevTime']*1000;
                spkTime = spkTime.astype(int);
                spkTime = spkTime[spkTime[:]<900];
                StimResp[i]['neurons'][j]['spkMtx'][r,spkTime] = 1;
                StimResp[i]['neurons'][j]['numspikes'][r] = np.sum(StimResp[i]['neurons'][j]['spkMtx'][r,TimeOfInterest]);                        
            
            StimResp[i]['neurons'][j]['meanSDF'] = makeSDF.getSDF(StimResp[i]['neurons'][j]['spkMtx'],1000);
            mResp[i,j] = np.mean(StimResp[i]['neurons'][j]['meanSDF'][TimeOfInterest])
    print('StimResp was made');             

    del experiment['stimStructs'];
    del experiment['iti_start'];
    del experiment['iti_end'];
    del experiment['iti'];
    experiment['filename'] = filename;
    experiment['StimResp'] = StimResp;

    ### save experiment (processed file)
    path_to_save = imec_filename[:(imec_filename.rfind('/')+1)] + 'processed/'; 
    if os.path.exists(path_to_save)==0:
        os.mkdir(path_to_save); 
    #name_to_save = path_to_save + bin_filename[(bin_filename.rfind('/')+1):-8] + 'npz';
    #np.savez_compressed(name_to_save, **experiment); 

    #name_to_save = path_to_save + bin_filename[(bin_filename.rfind('/')+1):-8] + 'json.gz';
    #f = gzip.GzipFile(name_to_save,'w');    
    #f.write(json.dumps(experiment, cls=NumpyEncoder).encode('utf-8')); 
    #f.close(); 
    #print('processed file was saved'); 


    ### stimulus condition information ###
    # No stim (1)
    # Center alone (2-4)
    # Surround alone (5-40)
    # CenterA + Surround (41-76)
    # CenterB + Surround (77-112)
    # CenterC + Surround (113-148)

    NRmargin = 1.2
    #%% Color for cAlone, sAlone
    NormResp1 = np.zeros((40,experiment['numNeurons']));
    for j in np.arange(experiment['numNeurons']):
        aloneMtx = mResp[:40,j];
        NormResp1[:,j] = aloneMtx-aloneMtx[0];
        #NormResp1[:,j] = NormResp1[:,j]/np.max(np.abs(NormResp1[:,j]));
        NormResp1[:,j] = NormResp1[:,j]/np.max(mResp[:,j]*NRmargin);    

    #%% Color for Cen+Surr
    NormRespA0 = np.zeros((36,experiment['numNeurons']));
    NormRespB0 = np.zeros((36,experiment['numNeurons']));
    NormRespC0 = np.zeros((36,experiment['numNeurons']));
    for j in np.arange(experiment['numNeurons']):
        aloneMtx = mResp[:40,j];    
        CenASurrMtx = mResp[40:76,j];
        CenBSurrMtx = mResp[76:112,j];
        CenCSurrMtx = mResp[112:148,j];    
        NormRespA0[:,j] = CenASurrMtx - aloneMtx[0];
        #NormRespA0[:,j] = NormRespA0[:,j]/np.max(np.abs(NormRespA0[:,j]));
        NormRespA0[:,j] = NormRespA0[:,j]/np.max(mResp[:,j]*NRmargin);    
        NormRespB0[:,j] = CenBSurrMtx - aloneMtx[0];
        #NormRespB0[:,j] = NormRespB0[:,j]/np.max(np.abs(NormRespB0[:,j]));
        NormRespB0[:,j] = NormRespB0[:,j]/np.max(mResp[:,j]*NRmargin);        
        NormRespC0[:,j] = CenCSurrMtx - aloneMtx[0];
        #NormRespC0[:,j] = NormRespC0[:,j]/np.max(np.abs(NormRespC0[:,j]));   
        NormRespC0[:,j] = NormRespC0[:,j]/np.max(mResp[:,j]*NRmargin);            

    #%% Color for surrMod: Cen+Surr - cAlone
    NormRespA1 = np.zeros((36,experiment['numNeurons']));
    NormRespB1 = np.zeros((36,experiment['numNeurons']));
    NormRespC1 = np.zeros((36,experiment['numNeurons']));
    for j in np.arange(experiment['numNeurons']):
        aloneMtx = mResp[:40,j];    
        CenASurrMtx = mResp[40:76,j];
        CenBSurrMtx = mResp[76:112,j];
        CenCSurrMtx = mResp[112:148,j];    
        NormRespA1[:,j] = CenASurrMtx - aloneMtx[1];
        #NormRespA1[:,j] = NormRespA1[:,j]/np.max(np.abs(NormRespA1[:,j]));
        NormRespA1[:,j] = NormRespA1[:,j]/np.max(mResp[:,j]*NRmargin);            
        NormRespB1[:,j] = CenBSurrMtx - aloneMtx[2];
        #NormRespB1[:,j] = NormRespB1[:,j]/np.max(np.abs(NormRespB1[:,j]));
        NormRespB1[:,j] = NormRespB1[:,j]/np.max(mResp[:,j]*NRmargin);            
        NormRespC1[:,j] = CenCSurrMtx - aloneMtx[3];
        #NormRespC1[:,j] = NormRespC1[:,j]/np.max(np.abs(NormRespC1[:,j]));
        NormRespC1[:,j] = NormRespC1[:,j]/np.max(mResp[:,j]*NRmargin);            
        
    #%% Color for surrMod: Cen+Surr - cAlone - sAlone    
    NormRespA2 = np.zeros((36,experiment['numNeurons']));
    NormRespB2 = np.zeros((36,experiment['numNeurons']));
    NormRespC2 = np.zeros((36,experiment['numNeurons']));
    for j in np.arange(experiment['numNeurons']):
        aloneMtx = mResp[:40,j];    
        CenASurrMtx = mResp[40:76,j];
        CenBSurrMtx = mResp[76:112,j];
        CenCSurrMtx = mResp[112:148,j];    
        NormRespA2[:,j] = CenASurrMtx - aloneMtx[1] - aloneMtx[4:];
        #NormRespA2[:,j] = NormRespA2[:,j]/np.max(np.abs(NormRespA2[:,j]));
        NormRespA2[:,j] = NormRespA2[:,j]/np.max(mResp[:,j]*NRmargin);            
        NormRespB2[:,j] = CenBSurrMtx - aloneMtx[2] - aloneMtx[4:];
        #NormRespB2[:,j] = NormRespB2[:,j]/np.max(np.abs(NormRespB2[:,j]));
        NormRespB2[:,j] = NormRespB2[:,j]/np.max(mResp[:,j]*NRmargin);                
        NormRespC2[:,j] = CenCSurrMtx - aloneMtx[3] - aloneMtx[4:];
        #NormRespC2[:,j] = NormRespC2[:,j]/np.max(np.abs(NormRespC2[:,j]));    
        NormRespC2[:,j] = NormRespC2[:,j]/np.max(mResp[:,j]*NRmargin);                

    #%% Drawing part
    import matplotlib.pyplot as plt;

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

    for jj in np.arange(nClusters):

        j = neurons_from_strong[jj]; 
        if app.all_radiobutton.isChecked() == True:
            unit_id = experiment['neuronid'][j]; 
        elif app.sua_radiobutton.isChecked() == True:
            unit_id = experiment['id_sua'][j]; 
        elif app.mua_radiobutton.isChecked() == True:
            unit_id = experiment['id_mua'][j]; 

        # Center Alone conditions
        for i in np.arange(3):
            plt.subplot(3,4,i+1);
            if NormResp1[i+1,j]>=0:
                cNow = np.array([1.0,1.0,1.0])-np.array([0.0,1.0,1.0])*NormResp1[i+1,j];
            else:
                cNow = np.array([1.0,1.0,1.0])-np.array([1.0,1.0,0.0])*abs(NormResp1[i+1,j]);
            plt.plot(0,0,'o',color=cNow,markersize=10);    
            plt.xlim([-2,2]);
            plt.ylim([-2,2]);        

        # Surround Alone conditions    
        plt.subplot(3,4,4);
        for i in np.arange(36):
            if i<6:
                cs_dist = 1;
                cs_dir = i*2*np.pi/6.0;
            elif i<18:
                cs_dist = 2;
                cs_dir = (i-6)*2*np.pi/12.0;
            else:
                cs_dist = 3;
                cs_dir = (i-18)*2*np.pi/18.0;
            xshift = 0.5*cs_dist*np.cos(cs_dir);
            yshift = 0.5*cs_dist*(-np.sin(cs_dir));        

            if NormResp1[i+4,j]>=0:
                cNow = np.array([1.0,1.0,1.0])-np.array([0.0,1.0,1.0])*NormResp1[i+4,j];
            else:
                cNow = np.array([1.0,1.0,1.0])-np.array([1.0,1.0,0.0])*abs(NormResp1[i+4,j]);
            plt.plot(0+xshift,0+yshift,'o',color=cNow,markersize=10);    
            plt.xlim([-2,2]);
            plt.ylim([-2,2]);        
                    
        # Cen+Surr conditions
        for i in np.arange(3):
            plt.subplot(3,4,5+i);
            plt.plot(0,0,'o',mec=[0,0,0],color=[1,1,1],markersize=10);            
            for p in np.arange(36):
                if p<6:
                    cs_dist = 1;
                    cs_dir = p*2*np.pi/6.0;
                elif p<18:
                    cs_dist = 2;
                    cs_dir = (p-6)*2*np.pi/12.0;
                else:
                    cs_dist = 3;
                    cs_dir = (p-18)*2*np.pi/18.0;
                xshift = 0.5*cs_dist*np.cos(cs_dir);
                yshift = 0.5*cs_dist*(-np.sin(cs_dir));   
                
                if i==0:
                    NormRespNow = NormRespA0;
                elif i==1:
                    NormRespNow = NormRespB0;
                else:
                    NormRespNow = NormRespC0;

                if NormRespNow[p,j]>=0:
                    cNow = np.array([1.0,1.0,1.0])-np.array([0.0,1.0,1.0])*NormRespNow[p,j];
                else:
                    cNow = np.array([1.0,1.0,1.0])-np.array([1.0,1.0,0.0])*abs(NormRespNow[p,j]);
                plt.plot(0+xshift,0+yshift,'o',color=cNow,markersize=10);    
                plt.xlim([-2,2]);
                plt.ylim([-2,2]);        
        """
        # surrMod conditions: Cen+Surr - cAlone 
        for i in np.arange(3):
            plt.subplot(3,4,9+i);
            for p in np.arange(36):
                if p<6:
                    cs_dist = 1;
                    cs_dir = p*2*np.pi/6.0;
                elif p<18:
                    cs_dist = 2;
                    cs_dir = (p-6)*2*np.pi/12.0;
                else:
                    cs_dist = 3;
                    cs_dir = (p-18)*2*np.pi/18.0;
                xshift = 0.5*cs_dist*np.cos(cs_dir);
                yshift = 0.5*cs_dist*(-np.sin(cs_dir));   
                
                if i==0:
                    NormRespNow = NormRespA1[:];
                elif i==1:
                    NormRespNow = NormRespB1[:];
                else:
                    NormRespNow = NormRespC1[:];

                if NormRespNow[p,j]>=0:
                    cNow = np.array([1.0,1.0,1.0])-np.array([0.0,1.0,1.0])*NormRespNow[p,j];
                else:
                    cNow = np.array([1.0,1.0,1.0])-np.array([1.0,1.0,0.0])*abs(NormRespNow[p,j]);
                plt.plot(0+xshift,0+yshift,'o',color=cNow,markersize=10);    
                plt.xlim([-2,2]);
                plt.ylim([-2,2]);        
        """

        # surrMod conditions: Cen+Surr - cAlone - sAlone
        for i in np.arange(3):
            plt.subplot(3,4,9+i);
            for p in np.arange(36):
                if p<6:
                    cs_dist = 1;
                    cs_dir = p*2*np.pi/6.0;
                elif p<18:
                    cs_dist = 2;
                    cs_dir = (p-6)*2*np.pi/12.0;
                else:
                    cs_dist = 3;
                    cs_dir = (p-18)*2*np.pi/18.0;
                xshift = 0.5*cs_dist*np.cos(cs_dir);
                yshift = 0.5*cs_dist*(-np.sin(cs_dir));   
                
                if i==0:
                    NormRespNow = NormRespA2[:];
                elif i==1:
                    NormRespNow = NormRespB2[:];
                else:
                    NormRespNow = NormRespC2[:];

                if NormRespNow[p,j]>=0:
                    cNow = np.array([1.0,1.0,1.0])-np.array([0.0,1.0,1.0])*NormRespNow[p,j];
                else:
                    cNow = np.array([1.0,1.0,1.0])-np.array([1.0,1.0,0.0])*abs(NormRespNow[p,j]);
                plt.plot(0+xshift,0+yshift,'o',color=cNow,markersize=10);    
                plt.xlim([-2,2]);
                plt.ylim([-2,2]);            
    #plt.show(block=True)

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


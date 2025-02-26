#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:38:14 2019

ExtractWaveform_NPX_Analysis.py

@author: taekjunkim
"""
#%%
import sys
import matplotlib.pyplot as plt
from phylib.io.model import load_model
#from phylib.utils.color import selected_cluster_color
import pandas as pd; 
import numpy as np; 
import json;
import gzip;
import os; 


#%%
def main(app):

    #%% path to params.py
    imec_filename = app.imec_file; 
    imec_dataFolder = imec_filename[:(imec_filename.rfind('/')+1)]; 
    params_path = imec_dataFolder + 'params.py'; 

    # first, we load the TemplateModel
    model = load_model(params_path); 

    # read cluster information
    man_sorted = app.sorted_checkbox.isChecked();     
    if man_sorted==1:
        ci_df = pd.read_csv(imec_dataFolder+'cluster_info.tsv', sep='\t');
    else:
        print('this should be processed after the manual sorting'); 
        return 0

    # for each cluster
    mean_wf = []; 
    for idx in range(np.shape(ci_df)[0]):

        if ci_df['group'][idx]=='good':

            if 'cluster_id' in ci_df.columns:
                cid = ci_df['cluster_id'][idx]; 
            else:
                cid = ci_df['id'][idx]; 

            # We get the waveforms of the cluster.        
            waveforms = model.get_cluster_spike_waveforms(cid); 
            try:
                n_spikes, n_samples, n_channels_loc = waveforms.shape; 
                # We get the channel ids where the waveforms are located.
                channel_ids = model.get_cluster_channels(cid); 
                wf_now = np.mean(waveforms[:,:,0],axis=0);   # mean_waveform from the best channel
            except:
                wf_now = np.zeros((82,)); 
        
            mean_wf.append(dict());     
            mean_wf[-1]['cluster_id'] = cid; 
            mean_wf[-1]['mean_wf'] = wf_now; 
            mean_wf[-1]['group'] = ci_df['group'][idx]; 
            mean_wf[-1]['ch'] = ci_df['ch'][idx];     
            mean_wf[-1]['depth'] = ci_df['depth'][idx];         
            print(f'cluster_id: {cid} was processed'); 

    # save data    
    path_to_save = imec_dataFolder + 'processed/'; 
    if os.path.exists(path_to_save)==0:
        os.mkdir(path_to_save); 
    name_to_save = imec_dataFolder + 'processed/mean_waveform.npz';    
    np.savez_compressed(name_to_save, **mean_wf)

    #name_to_save = imec_dataFolder + 'processed/mean_waveform.json.gz'; 
    #f = gzip.GzipFile(name_to_save,'w');    
    #f.write(json.dumps(mean_wf, cls=NumpyEncoder).encode('utf-8')); 
    #f.close(); 
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


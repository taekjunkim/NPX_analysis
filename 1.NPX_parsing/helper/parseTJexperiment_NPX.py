#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 18:33:18 2019

parse nex file using nexfile provided by "https://www.neuroexplorer.com/downloadspage/"
neo.io.NeuroExplorerIO has some errors
1) cannot load waveform
2) cannot read onset time of each analog signal fragment

@author: taekjunkim
"""
#%%
# readSGLX.py is useful to understand how to handle raw data
import numpy as np;
import os.path
import pandas as pd; 
import glob; 

#%%
def main(bin_filename, dat_filename, prevTime, numConds, imec_filename, app):

    ### spikets
    """
    neuronIDs and spike_timing from the sorted output
    """
    task_index_in_combine = int(app.tasknum_lineEdit.text()); 
    man_sorted = app.sorted_checkbox.isChecked(); 

    imec_dataFolder = imec_filename[:(imec_filename.rfind('/')+1)]; 

    if  (os.path.exists(imec_dataFolder+'../info/imec_datainfo.npy')) or (os.path.exists(imec_dataFolder+'info/imec_datainfo.npy')):
        try:
            imec_info = np.load(imec_dataFolder+'info/imec_datainfo.npy', allow_pickle=True).item();         
        except:
            imec_info = np.load(imec_dataFolder+'../info/imec_datainfo.npy', allow_pickle=True).item();         
        sync_start_end = dict(); 
        try: 
            sync_start_end['nidq'] = [imec_info['nidq: syncON'][task_index_in_combine], imec_info['nidq: syncOFF'][task_index_in_combine]]; 
            sync_start_end['ap_bin'] = [imec_info['ap: syncON'][task_index_in_combine], imec_info['ap: syncOFF'][task_index_in_combine]]; 
            sync_start_end['lf_bin'] = [imec_info['lf: syncON'][task_index_in_combine], imec_info['lf: syncOFF'][task_index_in_combine]];  
        except:
            sync_start_end['nidq'] = [imec_info['nidq_syncON'][task_index_in_combine], imec_info['nidq_syncOFF'][task_index_in_combine]]; 
            sync_start_end['ap_bin'] = [imec_info['ap_syncON'][task_index_in_combine], imec_info['ap_syncOFF'][task_index_in_combine]]; 
            sync_start_end['lf_bin'] = [imec_info['lf_syncON'][task_index_in_combine], imec_info['lf_syncOFF'][task_index_in_combine]];  
    else:
        sync_start_end = compute_syncONs(imec_filename); 
    
    id_all, spikets_all, chpos_all = get_spikeTS(imec_filename, task_index_in_combine, man_sorted, sync_start_end); 

    if app.sua_radiobutton.isChecked() == True:
        spikets = spikets_all['sua']; 
        neuronid = id_all['sua']; 
    elif app.mua_radiobutton.isChecked() == True:
        spikets = spikets_all['mua']; 
        neuronid = id_all['mua']; 
    elif app.all_radiobutton.isChecked() == True:
        spikets = spikets_all['sua'] + spikets_all['mua']; 
        neuronid = id_all['sua'] + id_all['mua']; 
        pass; 

    numNeurons = len(spikets); 


    #spikets = [[0,1,2,3]]; # fake spikets: 1 unit, 4 spikes
    #numNeurons = len(spikets); 
    #neuronid = 0; 

    ### markerts, markervals, photodiode
    """
    markerts: event_timing from a neuropixel bin file
        this can be done by reading digital ch of bin file
    markervals: event information from a dat file from PYPE
    pdOnTS, pdOffTS are extracted from a neuropixel bin file
        this can be done by reading digital ch of bin file
    markervals_str: this is not used, but shows actual meanings of event numbers
    """
    markervals, markervals_str = get_markervals(dat_filename); 

    markerts, pdOnTS, pdOffTS = get_event_ts(bin_filename, markervals_str, sync_start_end); 

    ### get parseParams
    parseParams = get_parseParams(); 
    
    ### get experiment parameters
    experiment = dict(); 
    postTime = prevTime; 
    experiment['iti_start'] = []; 
    experiment['iti_end'] = [];    
    experiment['numNeurons'] = numNeurons;        
    experiment['neuronid'] = neuronid;        
    experiment['id_sua'] = id_all['sua'];        
    experiment['id_mua'] = id_all['mua'];
    experiment['chpos_sua'] = chpos_all['sua'];        
    experiment['chpos_mua'] = chpos_all['mua'];
    experiment['prevTime'] = prevTime;                        
    experiment['postTime'] = postTime;          
    experiment['correct'] = [];                    

    counter = 0; 
    ### points where individual trials begin
    #stimITIOns = np.where(markervals==parseParams['startITICode'])[0];     
    stimITIOns = np.where(markervals_str=='start_iti')[0];     
    while (counter < stimITIOns[0]): 
        experiment[markervals_str[counter]] = markervals[counter+1]; 
        counter += 2; 

    ### Prepare StimStructs
    stimStructs = []; 
    for i in np.arange(numConds):
        stimStructs.append(dict()); 
        stimStructs[i]['numInstances'] = 0;
        stimStructs[i]['timeOn'] = []; 
        stimStructs[i]['timeOff'] = []; 
        stimStructs[i]['pdOn'] = []; 
        stimStructs[i]['pdOff'] = []; 
        stimStructs[i]['neurons'] = [];        
        stimStructs[i]['trial_num'] = [];                
        for j in np.arange(numNeurons):
            stimStructs[i]['neurons'].append(dict());

    ### Prepare to get stimulus information parameters                
    if stimITIOns[0] != counter:
        print('The first start_iti code is offset');
    stimOns = np.where(markervals==parseParams['stimOnCode'])[0];        
    
    error_indices = []; 
    completedITIs = 0; 
    ### Get stimuli
    for i in np.arange(len(stimITIOns)-1): # the file should end with 
                                           # a startITI that we don't care about    
        if stimITIOns[i] < counter:
            continue;

        index = stimITIOns[i] + 1; 
        next_code = markervals[index]; 

        if next_code == parseParams['endITICode']:
            experiment['iti_start'].append(markerts[stimITIOns[i]]);
            experiment['iti_end'].append(markerts[index]);
            completedITIs = completedITIs + 1;

        elif next_code == parseParams['pauseCode']:    
            if markervals[index+1] != parseParams['unpauseCode']:
                print('Found pause, but no unpause at '+str(index+1));
                print('continuing from next start_iti');                
                error_indices.append(index);           
                continue;
            index = index + 2; 
            next_code = markervals[index];

            if next_code == parseParams['endITICode']:
                experiment['iti_start'].append(markerts[stimITIOns[i]]);
                experiment['iti_end'].append(markerts[index]);
                completedITIs = completedITIs + 1;
            else:
                print('Found bad code '+str(next_code)+' after start_iti at index '+str(index));
                print('continuing from next start_iti');                
                error_indices.append(index);           
                continue;

        else:                
            print('Found bad code '+str(next_code)+' after start_iti at index '+str(index));
            print('continuing from next start_iti');                
            error_indices.append(index);           
            continue;

        next_code2 = markervals[index+1]; 
        if next_code2 == parseParams['fixAcquiredCode']:
            pass; 
        elif next_code2 == parseParams['UninitiatedTrialCode']:
            if markervals[index+2] != parseParams['startITICode']:
                error_indices.append(index+2);
                print('Found non start_iti code '+str(markervals[index+2])+
                      ' after Uninitiated trial at '+str(index+2));
            continue;
        else:
            print('Found bad code '+str(next_code2)+' after end_iti at index '
                  +str(stimITIOns[i]+2));
            error_indices.append(index);           
            continue;

        ndex = index + 2;
        trialCode = [];

        while (trialCode == []):        
            optionalCode = 0; 
            stimCode = markervals[ndex+optionalCode];
            
            if stimCode == parseParams['fixLost']:
                if hasValidBreakFix(ndex+optionalCode,markervals,parseParams):
                    trialCode = parseParams['breakFixCode'];
                    continue;
            elif stimCode != parseParams['stimIDCode']:
                print('Found '+str(stimCode)+' as a stimID or breakfix code at stim time '
                      +str(markerts[ndex+optionalCode])+' at index '+str(ndex+optionalCode));
                print('continuing from next start_iti');                            
                error_indices.append(ndex+optionalCode);
                trialCode = parseParams['codeError'];
                continue;
                
            if markervals[ndex+1+optionalCode] == parseParams['fixLost']:
                if hasValidBreakFix(ndex+1+optionalCode,markervals,parseParams):
                    trialCode = parseParams['breakFixCode'];
                    continue;
            elif ((markervals[ndex+1+optionalCode] >= parseParams['stimIDOffset']) 
                  and (markervals[ndex+1+optionalCode] < parseParams['stimRotOffset'])): 
                stimIDCodeToStore = markervals[ndex+1+optionalCode]; 
            else:
                print('Found '+str(markervals[ndex+1])+' as a stimulus code at stim time '
                      +str(markerts[ndex+1+optionalCode])+' at index '+str(ndex+1+optionalCode)); 
                print('continuing from next start_iti');                            
                error_indices.append(ndex+optionalCode+1); 
                trialCode = parseParams['codeError']; 
                continue;                

            ## next code is either fixlost or stimOn
            codeIndex = ndex + 2 + optionalCode; 
            code = markervals[codeIndex];
            if code == parseParams['fixLost']:
                if hasValidBreakFix(codeIndex,markervals,parseParams):
                    trialCode = parseParams['breakFixCode'];
                    continue;
            elif code != parseParams['stimOnCode']:          
                print('Missing StimOn or fixlost code, found '+str(code)+' at '+str(codeIndex))
                print('continuing from next start_iti');                                        
                error_indices.append(codeIndex); 
                trialCode = parseParams['codeError']; 
                continue;                
            else:
                stimOnTime = markerts[codeIndex]; 
                
            ## next code is either fixlost or stimOff
            codeIndex = ndex + 3 + optionalCode; 
            code = markervals[codeIndex];
            if code == parseParams['fixLost']:
                if hasValidBreakFix(codeIndex,markervals,parseParams):
                    trialCode = parseParams['breakFixCode'];
                    continue;
            elif code != parseParams['stimOffCode']:          
                print('Missing StimOff or fixlost code, found '+str(code)+' at '+str(codeIndex))
                print('continuing from next start_iti');                                        
                error_indices.append(codeIndex);
                trialCode = parseParams['codeError'];
                continue;                
            else:
                stimOffTime = markerts[codeIndex]; 
                
            ## having made it here, we can now call this a completed stimulus presentation and record the results                
            sIndex = stimIDCodeToStore - parseParams['stimIDOffset']; 
            sIndex = sIndex - 1; # for zero-based indexing in python (Matlab doesn't need this);
            if stimStructs[sIndex]['numInstances'] == []:
                stimStructs[sIndex]['numInstances'] = 1; 
            else:                    
                stimStructs[sIndex]['numInstances'] = stimStructs[sIndex]['numInstances'] + 1;
                
            inst = stimStructs[sIndex]['numInstances'];
            inst = inst - 1; # for zero-based indexing in python (Matlab doesn't need this);
            stimStructs[sIndex]['timeOn'].append(stimOnTime);               
            stimStructs[sIndex]['timeOff'].append(stimOffTime);               

            ## add trial num                
            stimStructs[sIndex]['trial_num'].append(i);                           

            ## now find the pdiode events associated with
            pdOnsAfter = np.where(pdOnTS > stimOnTime)[0]; 
            if len(pdOnsAfter)==0:
                print('Error, did not find a photodiode on code after stimon at time '+str(stimOnTime));
                print('Ignoring... Continuing');
            else:
                pdOffsAfter = np.where(pdOffTS > pdOnTS[pdOnsAfter[0]])[0];
                if len(pdOffsAfter)==0:
                    print('Error, did not find a photodiode on code after stimon at time '+str(pdOnTS[pdOnsAfter[0]]));
                    print('Ignoring... Continuing');
                else:
                    stimStructs[sIndex]['pdOn'].append(pdOnTS[pdOnsAfter[0]]);
                    stimStructs[sIndex]['pdOff'].append(pdOffTS[pdOffsAfter[0]]);                    

            ## now get neural data
            for j in np.arange(numNeurons):
                mySpikes = np.array([]);
                if stimStructs[sIndex]['pdOff'] != []:
                    spikeIndices = np.where((spikets[j] >= (stimStructs[sIndex]['pdOn'][inst]-prevTime)) & 
                                            (spikets[j] <= (stimStructs[sIndex]['pdOff'][inst]+postTime)))[0];
                else:
                    spikeIndices = np.where((spikets[j] >= (stimOnTime-prevTime)) & 
                                            (spikets[j] <= (stimOffTime+postTime)))[0];
                                            
                if len(spikeIndices)>0:
                    mySpikes = np.append(mySpikes,spikets[j][spikeIndices]);
                if inst == 0:
                    stimStructs[sIndex]['neurons'][j]['spikes'] = [];
                stimStructs[sIndex]['neurons'][j]['spikes'].append(mySpikes);    
                
            ## next code is either fixlost, an object code or correct_response
            codeIndex = ndex + 4 + optionalCode;
            code = markervals[codeIndex];

            if code == parseParams['fixLost']:
                if hasValidBreakFix(codeIndex,markervals,parseParams):
                    trialCode = parseParams['breakFixCode'];
                    continue;
            elif code == parseParams['correctCode']: # end of trial
                experiment['correct'].append(i);                                     
                if markervals[codeIndex+1] != parseParams['startITICode']:
                    print('Missing startITI after '+str(markervals[codeIndex+1])+
                          ' at '+str(markerts[codeIndex+1])+' at index '+
                          str(codeIndex+1));
                    error_indices.append(codeIndex);
                    trialCode = parseParams['codeError'];
                    continue;                
                else:
                    trialCode = parseParams['correctCode'];
                    continue;
            elif code != parseParams['stimIDCode']:                
                print('Found '+str(stimCode)+' as a stim ID code at stim time '+
                      str(markerts[codeIndex]));
                print('continuing from next start_iti');                        
                error_indices.append(codeIndex);
                trialCode = parseParams['codeError'];
                continue;                
            else:
                ndex = ndex + 4 + optionalCode;                    
                                    
    ### add stimStructs to experiment output, then return
    experiment['stimStructs'] = stimStructs;                        
    experiment['errors'] = error_indices; 
    
    return experiment; 

#%% hasValidBreakFix
def hasValidBreakFix(ndex, markervals, parseParams):
    if markervals[ndex+1] != parseParams['breakFixCode']:
        print('missing breakFixCode after '+str(markervals[ndex])+
              ' at index '+str(ndex));
    if markervals[ndex+2] != parseParams['startITICode']:
        print('missing startITI after '+str(markervals[ndex+1])+
              ' at index '+str(ndex));
    yesno = 1;    
    return yesno;

#%% get_markervals
def get_markervals(dat_filename):
    
    # read dat_file and make markervals_str
    fid = open(dat_filename, 'r'); 
    markervals_str = []; 
    while True:
        tline = fid.readline(); 
        if tline=='':
            break; 
        # remove brackets, single quote. then make a single array
        markervals_str += tline[1:-2].replace("'",'').split(', '); 

    # make markervals by replacing markervals_str into numbers 
    markervals = []; 
    for idx, x in enumerate(markervals_str):
        if x=='color':
            markervals.append(42);
        elif x=='rfx':
            markervals.append(43);
        elif x=='rfy':
            markervals.append(44);
        elif x=='iti':
            markervals.append(45);
        elif x=='stim_time':
            markervals.append(46);
        elif x=='isi':
            markervals.append(47);
        elif x=='numstim':
            markervals.append(48);
        elif x=='stimid':
            markervals.append(49);
        elif x=='start':
            markervals.append(10);
        elif x=='stop':
            markervals.append(11);
        elif x=='start_iti':
            markervals.append(12);
        elif x=='end_iti':
            markervals.append(13);
        elif x=='eye_start':
            markervals.append(14);
        elif x=='eye_stop':
            markervals.append(15);
        elif x=='fix_on':
            markervals.append(29);
        elif x=='fix_off':
            markervals.append(30);
        elif x=='fix_acquired':
            markervals.append(31);
        elif x=='fix_lost':
            markervals.append(33);
        elif x=='reward':
            markervals.append(37);
        elif x=='sample_on':
            markervals.append(38);
        elif x=='sample_off':
            markervals.append(39);
        elif x=='C':
            markervals.append(0);
        elif x=='U':
            markervals.append(2);
        elif x=='B':
            markervals.append(6);
        elif x=='W':
            markervals.append(7);
        elif x=='N':
            markervals.append(8);
        elif x=='pause':
            markervals.append(100);
        elif x=='unpause':
            markervals.append(101);
        elif x=='rotid':
            markervals.append(50);
        elif x=='occlshape':
            markervals.append(62);
        elif x=='mask_info':
            markervals.append(54);
        elif x=='occlmode':
            markervals.append(52);
        elif x=='extra':
            markervals.append(74);
        elif x=='dot_rad':
            markervals.append(63);
        elif x=='occl_info':
            markervals.append(53);
        elif x=='ambiguous_info':
            markervals.append(75);
        elif x=='mon_ppd':
            markervals.append(78);
        elif x=='fix_x':
            markervals.append(76);
        elif x=='fix_y':
            markervals.append(77);
        else:
            try:
                markervals.append(int(float(x)));   
            except:      
                markervals.append(999);   
                print(idx, x); 
                print('Unknown event code was found');

    return np.array(markervals), np.array(markervals_str); 

#%% markerts, pdOnTS, pdOffTS
def get_event_ts(bin_filename, markervals_str, sync_start_end):
   
    meta_filename = bin_filename[:-3]+'meta'; 
    metaDict = get_metaDict(meta_filename); 

    rawData = access_rawData(bin_filename, metaDict); 
    ### rawData shape = (nChan, nFileSamp)
    ### nidq.bin file of Rig3
    ### ch idx0: eyeH
    ### ch idx1: eyeV
    ### ch idx2: sync - square wave going to nidq.bin & ap.bin
    ### ch idx3: pupil
    ### ch idx4: photodiode
    ### ch idx5: digital. photodiode (pin0), event (pin1)

    ### detect syncOn in the first 20 seconds
    niSampRate = int(metaDict['niSampRate']); 
    #syncCh = int(metaDict['syncNiChan']); 
    #syncONs = np.where(rawData[syncCh,:niSampRate*20]
    #                  >np.max(rawData[syncCh,:niSampRate*20])*0.5)[0];    
    #for p in range(10):
    #    if syncONs[p+10]-syncONs[p]==10:
    #        syncON = syncONs[p]; 
    #        break; 

    syncON = sync_start_end['nidq'][0]; 

    ### read digit signal
    digitCh = np.shape(rawData)[0]-1;   # the last channel     
    digit_signal = rawData[digitCh, :]; 
    digit_diff = digit_signal[1:] - digit_signal[:-1]; 

    ### time (ms) with respect to syncON
    markerts = (np.where(digit_diff==2)[0] + 1 - syncON)/niSampRate; 
    """
    pdOnTS_raw = (np.where(digit_diff==1)[0] + 1 - syncON)/niSampRate; 
    pdOffTS_raw = (np.where(digit_diff==-1)[0] + 1 - syncON)/niSampRate; 

    ### this is for photodiode generating pdOn for every frame
    pdOn_dist = pdOnTS_raw[1:] - pdOnTS_raw[:-1];
    pdOnTS = np.append(pdOnTS_raw[0],
                       pdOnTS_raw[np.where(pdOn_dist>0.02)[0]+1]);

    pdOff_dist = pdOffTS_raw[1:] - pdOffTS_raw[:-1];
    pdOffTS = np.append(pdOffTS_raw[np.where(pdOff_dist>0.02)[0]],
                        pdOffTS_raw[-1]); 
    """
    """
    ### this is for photodiode generating continuous pdOn
    pdOnTS = pdOnTS_raw; 
    pdOffTS = pdOffTS_raw; 
    """

    ### photodiode based on analog signal
    pd_signal = rawData[4, :]; 
    pd_signal = pd_signal - np.max([0, np.min(pd_signal)]); 
    pd_High = np.where(pd_signal > np.max(pd_signal)*0.5)[0]; 
    pdOnTS = np.concatenate(([pd_High[0]], pd_High[np.where(np.diff(pd_High)>25*5)[0]+1]));     
    pdOffTS = np.concatenate((pd_High[np.where(np.diff(pd_High)>25*5)[0]], [pd_High[-1]])); 
    pdOnTS = (pdOnTS - syncON)/niSampRate; 
    pdOffTS = (pdOffTS - syncON)/niSampRate; 


    if len(markerts)==len(markervals_str):
        print('Good: number of events are well matched');         
        pass; 
    else:
        markerts = np.zeros(len(markervals_str)); 
        stim_ons = np.where(markervals_str=='sample_on')[0]; 
        #stim_ons = np.where(markervals_str=='test_on')[0];         
        if len(stim_ons) == len(pdOnTS):
            print('number of pdONs is matched with "sample_on"');             
            markerts[stim_ons] = pdOnTS - 0.001;   # 1 ms earlier than pdOnTS  
        else:
            print('number of pdONs is not matched with "sample_on"'); 
            if len(stim_ons)<len(pdOnTS):
                markerts[stim_ons] = pdOnTS[:len(stim_ons)] - 0.001; 
            else:
                markerts[stim_ons[:len(pdOnTS)]] = pdOnTS - 0.001; 

    # timestamps in seconds
    return markerts, pdOnTS, pdOffTS

#%% get_metaDict
def get_metaDict(metaname):
    metaDict = {}
    with open(metaname) as f:
        mdatList = f.read().splitlines()
        # convert the list entries into key value pairs
        for m in mdatList:
            csList = m.split(sep='=')
            if csList[0][0] == '~':
                currKey = csList[0][1:len(csList[0])]
            else:
                currKey = csList[0]
            metaDict.update({currKey: csList[1]})
    return metaDict;

#%% get_spikets
def get_spikeTS(imec_filename, task_index_in_combine, man_sorted, sync_start_end):    

    imec_dataFolder = imec_filename[:(imec_filename.rfind('/')+1)]; 

    if (os.path.exists(imec_dataFolder+'../info/imec_datainfo.npy')) or (os.path.exists(imec_dataFolder+'info/imec_datainfo.npy')):
        try:
            imec_info = np.load(imec_dataFolder+'info/imec_datainfo.npy', allow_pickle=True).item(); 
        except:
            imec_info = np.load(imec_dataFolder+'../info/imec_datainfo.npy', allow_pickle=True).item(); 
        try:
            imSampRate = imec_info['ap: imSampRate'][task_index_in_combine]; 
            #rawdata = []; #AC commented this out
            #binname = []; 
            syncON = imec_info['ap: firstSamp'][task_index_in_combine] + imec_info['ap: syncON'][task_index_in_combine]; 

            if 'ap: syncOFF' in imec_info.keys():
                syncDur_nidq = imec_info['nidq: syncOFF'][task_index_in_combine] - imec_info['nidq: syncON'][task_index_in_combine]; 
                syncDur_nidq_sec = syncDur_nidq / imec_info['nidq: SampRate'][task_index_in_combine]; 

                # adjusted imSampRate
                syncDur_ap = imec_info['ap: syncOFF'][task_index_in_combine] - imec_info['ap: syncON'][task_index_in_combine]; 
                imSampRate = syncDur_ap / syncDur_nidq_sec; 
        except:
            imSampRate = imec_info['ap_imSampRate'][task_index_in_combine]; 
            #rawdata = []; #AC commented this out
            #binname = []; 
            syncON = imec_info['ap_firstSamp'][task_index_in_combine] + imec_info['ap_syncON'][task_index_in_combine]; 

            if 'ap_syncOFF' in imec_info.keys():
                syncDur_nidq = imec_info['nidq_syncOFF'][task_index_in_combine] - imec_info['nidq_syncON'][task_index_in_combine]; 
                syncDur_nidq_sec = syncDur_nidq / imec_info['nidq_SampRate'][task_index_in_combine]; 

                # adjusted imSampRate
                syncDur_ap = imec_info['ap_syncOFF'][task_index_in_combine] - imec_info['ap_syncON'][task_index_in_combine]; 
                imSampRate = syncDur_ap / syncDur_nidq_sec; 

    else:
        ap_binname = imec_filename; 

        nidq_sync_dur = (sync_start_end['nidq'][1]-sync_start_end['nidq'][0])/25000; 
        imSampRate = (sync_start_end['ap_bin'][1]-sync_start_end['ap_bin'][0]) / nidq_sync_dur; 
        syncON = sync_start_end['ap_bin'][0];  
 
    st = np.load(imec_dataFolder+'spike_times.npy'); 
    sc = np.load(imec_dataFolder+'spike_clusters.npy'); 
    
    st = (st.astype(dtype='int64')-syncON)/imSampRate; 

    if man_sorted==1:
        df = pd.read_csv(imec_dataFolder+'cluster_info.tsv', sep='\t');
        ch_map = np.load(imec_dataFolder+'channel_map.npy').flatten();  
        ch_pos = np.load(imec_dataFolder+'channel_positions.npy');  
        for i in np.arange(df.shape[0]):
            id_chmap = np.where(ch_map==df.loc[i,'ch'])[0]; 
            df.loc[i,'xc'] = ch_pos[id_chmap,0]; 
            df.loc[i,'yc'] = ch_pos[id_chmap,1]; 
    else:
        df = pd.read_csv(imec_dataFolder+'cluster_KSLabel.tsv', sep='\t');

        ch_map = np.load(imec_dataFolder+'channel_map.npy').flatten();  
        ch_pos = np.load(imec_dataFolder+'channel_positions.npy');  

        templates = np.load(imec_dataFolder+'templates.npy'); 
        chan_best = (templates**2).sum(axis=1).argmax(axis=-1); 

        """
        spikeTemps = np.load(imec_dataFolder+'spike_templates.npy'); 
        spikeTempAmps = np.load(imec_dataFolder+'amplitudes.npy'); 
        pcFeat = np.load(imec_dataFolder+'pc_features.npy');     
        pcFeat = np.squeeze(pcFeat[:,0,:]);    # take first PC only
        pcFeat[pcFeat<0] = 0;    # some entries are negative, but we don't really want to push the CoM away from there.

        pcFeatInd = np.load(imec_dataFolder+'pc_feature_ind.npy');       
        ch_pos = np.load(imec_dataFolder+'channel_positions.npy');  
        ycoords = ch_pos[:,1]; 

        spikeFeatInd = pcFeatInd[spikeTemps,:];     
        spikeFeatYcoords = np.squeeze(ycoords[spikeFeatInd]);  # 2D matrix of size #spikes x 12 
        spikeDepths = np.sum(np.multiply(spikeFeatYcoords,pcFeat**2),axis=1)/np.sum(pcFeat**2,axis=1);  
        """

        for i in np.arange(df.shape[0]):
            spk_now = np.where(sc==df.loc[i,'cluster_id'])[0]; 
            ch_now = chan_best[i]; 

            df.loc[i,'xc'] = ch_pos[ch_now,0]; 
            df.loc[i,'yc'] = ch_pos[ch_now,1]; 

    
    ### check inter-spike interval
    ### np.diff, 5% rule

    id_all = dict(); 
    id_all['sua'] = []; 
    id_all['mua'] = []; 

    spikets_all = dict(); 
    spikets_all['sua'] = []; 
    spikets_all['mua'] = []; 

    chpos_all = dict(); 
    chpos_all['sua'] = []; 
    chpos_all['mua'] = []; 

    for i in np.arange(df.shape[0]):
        if 'cluster_id' in df.columns:
            cid = df.loc[i,'cluster_id'];
        else:
            cid = df.loc[i,'id'];
        spk_ts = st[sc[:]==cid];
        chpos = df.loc[i,['xc', 'yc']].to_numpy()
        
        if man_sorted==1:
            if df.loc[i,'group']=='good':
                id_all['sua'].append(cid);
                spikets_all['sua'].append(spk_ts);      
                chpos_all['sua'].append(chpos); 
            elif df.loc[i,'group']=='mua':
                id_all['mua'].append(cid);        
                spikets_all['mua'].append(spk_ts);
                chpos_all['mua'].append(chpos);     
        else:
            if df.loc[i,'KSLabel']=='good':
                id_all['sua'].append(cid);
                spikets_all['sua'].append(spk_ts);
                chpos_all['sua'].append(chpos); 
            elif df.loc[i,'KSLabel']=='mua':
                id_all['mua'].append(cid);        
                spikets_all['mua'].append(spk_ts);    
                chpos_all['mua'].append(chpos); 
    
    #del rawdata, binname; #AC commented this out

    # timestamps in seconds    
    return id_all, spikets_all, chpos_all; 

#%% compute_syncONs
def compute_syncONs(imec_filename, ap_pass=False):
    ap_binname = imec_filename[:-6]+'ap.bin'; 

    ### check nidq
    idx_slash = []; 
    for c in np.arange(len(ap_binname)):
        if ap_binname[c]=='/':
            idx_slash.append(c); 
    nidq_binname = glob.glob(ap_binname[:idx_slash[-2]]+'/*nidq.bin')[0]

    nidq_meta = get_metaDict(nidq_binname[:-3]+'meta'); 
    nidq_syncCH = int(nidq_meta['syncNiChan'])
    nidq_nChan = int(nidq_meta['nSavedChans']); 
    nidq_nFileSamp = int(int(nidq_meta['fileSizeBytes'])/(2*nidq_nChan)); 
    nidq_SampRate = int(nidq_meta['niSampRate']); 

    nidq_data = np.memmap(nidq_binname, dtype='int16', 
                        shape=(nidq_nFileSamp, nidq_nChan), offset=0, order='C'); 
    nidq_sync = nidq_data[:,nidq_syncCH].copy(); 
    nidq_sHigh = np.where(nidq_sync>10000)[0]; 
    nidq_sOFF_pre = np.concatenate((nidq_sHigh[np.where(np.diff(nidq_sHigh)>10)[0]], [nidq_sHigh[-1]])); 
    nidq_sON_pre = np.concatenate(([nidq_sHigh[0]], nidq_sHigh[np.where(np.diff(nidq_sHigh)>10)[0]+1])); 

    nidq_sON = []; 
    for t in np.arange(len(nidq_sON_pre)):
        if np.min(nidq_data[nidq_sON_pre[t]:nidq_sON_pre[t]+25,nidq_syncCH])>10000:
            nidq_sON.append(nidq_sON_pre[t]); 
    nidq_sON = np.array(nidq_sON); 
    nidq_sOFF = []; 
    for t in np.arange(len(nidq_sOFF_pre)):
        if np.min(nidq_data[nidq_sOFF_pre[t]-25:nidq_sOFF_pre[t],nidq_syncCH])>10000:
            nidq_sOFF.append(nidq_sOFF_pre[t]); 
    nidq_sOFF = np.array(nidq_sOFF); 

    print('NIDQ syncON/OFF: ',len(nidq_sON),len(nidq_sOFF)); 

    nidq_sync_good = np.array([0, 0]); 
    if nidq_sON[0] > np.max(np.diff(nidq_sHigh)): 
        nidq_sync_good[0] = 1; 
        print('NIDQ first syncON is OK'); 
    if nidq_nFileSamp - nidq_sOFF[-1] > np.max(np.diff(nidq_sHigh)): 
        nidq_sync_good[1] = 1;         
        print('NIDQ last syncOFF is OK'); 

    ### check lf.bin
    lf_binname = ap_binname[:-6]+'lf.bin'; 

    lf_meta = get_metaDict(lf_binname[:-3]+'meta'); 
    lf_nChan = int(lf_meta['nSavedChans']); 
    lf_nFileSamp = int(int(lf_meta['fileSizeBytes'])/(2*lf_nChan)); 
    lf_imSampRate = int(lf_meta['imSampRate']);        

    lf_data = np.memmap(lf_binname, dtype='int16', 
                        shape=(lf_nFileSamp, lf_nChan), offset=0, mode='r',order='C'); 
    lf_sync = lf_data[:,384].copy(); 
    del lf_data; 

    lf_sHigh = np.where(lf_sync==64)[0]; 
    lf_sON_pre = np.concatenate(([lf_sHigh[0]], lf_sHigh[np.where(np.diff(lf_sHigh)>10)[0]+1])); 
    lf_sOFF_pre = np.concatenate((lf_sHigh[np.where(np.diff(lf_sHigh)>10)[0]], [lf_sHigh[-1]])); 

    lf_sON = []; 
    for t in np.arange(len(lf_sON_pre)):
        if np.min(lf_sync[lf_sON_pre[t]:lf_sON_pre[t]+5])>0:
            lf_sON.append(lf_sON_pre[t]); 
    lf_sON = np.array(lf_sON); 
    lf_sOFF = []; 
    for t in np.arange(len(lf_sOFF_pre)):
        if np.min(lf_sync[lf_sOFF_pre[t]-5:lf_sOFF_pre[t]])>0:
            lf_sOFF.append(lf_sOFF_pre[t]); 
    lf_sOFF = np.array(lf_sOFF); 

    if lf_sON[0]==0:
        lf_sON = lf_sON[1:]; 
        lf_sOFF = lf_sOFF[1:]; 

    print('LF syncON/OFF: ',len(lf_sON),len(lf_sOFF)); 
    lf_sync_good = np.array([0, 0]);     
    if lf_sON[0] > np.max(np.diff(lf_sHigh)): 
        lf_sync_good[0] = 1;     
        print('LF first syncON is OK'); 
    if lf_nFileSamp - lf_sOFF[-1] > np.max(np.diff(lf_sHigh)): 
        lf_sync_good[1] = 1;             
        print('LF last syncOFF is OK'); 


    ### check ap.bin
    sON_valid_idx0 = len(nidq_sON) - len(lf_sON); 
    if sON_valid_idx0 < 0:   # this is sync error. weird signal in the middle
        if (np.min(nidq_sync_good)==1) and (np.min(lf_sync_good)==1):
            sON_valid_idx0 = 0; 
        else:
            sync_start_end = dict(); 
            sync_start_end['nidq'] = np.array([nidq_sON[0], nidq_sOFF[-1]]); 
            sync_start_end['ap_bin'] = np.array([np.nan, ap_sOFF]); 
            sync_start_end['lf_bin'] = np.array([np.nan, lf_sOFF[-1]]); 
            print('Found sync error!'); 
            return sync_start_end; 

    nidq_sync_dur = (nidq_sOFF[-1]-nidq_sON[sON_valid_idx0])/nidq_SampRate; 

    last_seconds = (lf_nFileSamp-lf_sOFF[-1])/lf_imSampRate; 
    last_seconds = int(last_seconds+1); 

    if ap_pass==False:
        ap_meta = get_metaDict(ap_binname[:-3]+'meta'); 
        ap_nChan = int(ap_meta['nSavedChans']); 
        ap_nFileSamp = int(int(ap_meta['fileSizeBytes'])/(2*ap_nChan)); 
        ap_imSampRate = int(ap_meta['imSampRate']); 

        ap_data = np.memmap(ap_binname, dtype='int16', 
                            shape=(ap_nFileSamp, ap_nChan), offset=0, mode='r',order='C'); 
        ap_sHigh_start = np.where(ap_data[:int(ap_imSampRate*10),384]==64)[0]; 
        ap_sONs_pre = np.concatenate(([ap_sHigh_start[0]], ap_sHigh_start[np.where(np.diff(ap_sHigh_start)>10)[0]+1])); 
        ap_sONs = []; 
        for t in np.arange(len(ap_sONs_pre)):
            if np.min(ap_data[ap_sONs_pre[t]:ap_sONs_pre[t]+30,384])>0:
                ap_sONs.append(ap_sONs_pre[t]); 
        ap_sONs = np.array(ap_sONs); 

        if ap_sONs[0]==0:
            ap_sON = ap_sONs[1]; 
        else:
            ap_sON = ap_sONs[0]; 

        ap_sHigh_end = np.where(ap_data[-int(ap_imSampRate*last_seconds):,384]==64)[0]; 
        ap_sOFFs_pre = ap_sHigh_end + ap_nFileSamp - ap_imSampRate*last_seconds;    
        ap_sOFFs = []; 
        for t in np.arange(len(ap_sOFFs_pre)):
            if np.min(ap_data[ap_sOFFs_pre[t]-30:ap_sOFFs_pre[t],384])>0:
                ap_sOFFs.append(ap_sOFFs_pre[t]); 
        ap_sOFF = ap_sOFFs[-1]; 
        imSampRate = (ap_sOFF - ap_sON) / nidq_sync_dur; 
        syncON = ap_sON;     

        sync_start_end = dict(); 
        sync_start_end['nidq'] = np.array([nidq_sON[sON_valid_idx0], nidq_sOFF[-1]]); 
        sync_start_end['ap_bin'] = np.array([ap_sON, ap_sOFF]); 
        sync_start_end['lf_bin'] = np.array([lf_sON[0], lf_sOFF[-1]]); 
    
    else:
        sync_start_end = dict(); 
        sync_start_end['nidq'] = np.array([nidq_sON[sON_valid_idx0], nidq_sOFF[-1]]); 
        sync_start_end['ap_bin'] = np.array([]); 
        sync_start_end['lf_bin'] = np.array([lf_sON[0], lf_sOFF[-1]]); 


    return sync_start_end; 



#%% get_rawData
def access_rawData(binFullPath, meta):  
    nChan = int(meta['nSavedChans'])
    nFileSamp = int(int(meta['fileSizeBytes'])/(2*nChan))
    print("nChan: %d, nFileSamp: %d" % (nChan, nFileSamp))
    rawData = np.memmap(binFullPath, dtype='int16', mode='r',
                        shape=(nChan, nFileSamp), offset=0, order='F')
    return(rawData)


#%% Load parameters, which will be compared with markervals
def get_parseParams():    
    parseParams = {};
    parseParams['add_extra_isiCode'] = 67;
    parseParams['background_infoCode'] = 70;
    parseParams['bar_downCode'] = 26;
    parseParams['bar_upCode'] = 25;
    parseParams['blackRespIndex'] = 101;
    parseParams['blank'] = 1;
    parseParams['breakFixCode'] = 6;
    parseParams['codeError'] = -1;
    parseParams['colorCode'] = 42;
    parseParams['correctCode'] = 0;
    parseParams['distanceThreshold'] = 10;
    parseParams['dot_radCode'] = 63;
    parseParams['EARLY_RELEASECode'] = 4;
    parseParams['end_post_trialCode'] = 19;
    parseParams['end_pre_trialCode'] = 17;
    parseParams['end_wait_barCode'] = 24;
    parseParams['end_wait_fixationCode'] = 21;
    parseParams['endITICode'] = 13;
    parseParams['extraCode'] = 74;
    parseParams['eye_startCode'] = 14;
    parseParams['eye_stopCode'] = 15;
    parseParams['fix_doneCode'] = 34;
    parseParams['fix_offCode'] = 30;
    parseParams['fix_onCode'] = 29;
    parseParams['fixAcquiredCode'] = 31;
    parseParams['fixation_occursCode'] = 22;
    parseParams['fixLost'] = 33;
    parseParams['foreground_infoCode'] = 69;
    parseParams['gen_modeCode'] = 65;
    parseParams['gen_submodeCode'] = 66;
    parseParams['isiCode'] = 47;
    parseParams['itiCode'] = 45;
    parseParams['LATE_RESPCode'] = 5;
    parseParams['line_widthCode'] = 64;
    parseParams['location_flip_infoCode'] = 73;
    parseParams['mask_infoCode'] = 54;
    parseParams['mask_offCode'] = 56;
    parseParams['mask_onCode'] = 55;
    parseParams['maxCode'] = 4095;
    parseParams['maxColorValue'] = 256;
    parseParams['MAXRT_EXCEEDEDCode'] = 3;
    parseParams['midground_infoCode'] = 68;
    parseParams['NO_RESPCode'] = 8;
    parseParams['occl_infoCode'] = 53;
    parseParams['occlmodeCode'] = 52;
    parseParams['occlshapeCode'] = 62;
    parseParams['OneBasedIndexing'] = 1;
    parseParams['onset_timeCode'] = 71;
    parseParams['pauseCode'] = 100;
    parseParams['pdiodeChannel'] = 'Event002';
    parseParams['pdiodeDistanceThreshold'] = 0.02;
    parseParams['pdiodeThresh'] = 4.8;
    parseParams['perispaceCode'] = 61;
    parseParams['plexFloatMultCode'] = 1000;
    parseParams['plexYOffsetCode'] = 600;
    parseParams['positionCode'] = 57;
    parseParams['radius_code'] = 80;
    parseParams['rewardCode'] = 37;
    parseParams['rfxCode'] = 43;
    parseParams['rfyCode'] = 44;
    parseParams['rotIDCode'] = 50;
    parseParams['second_stimuliCode'] = 72;
    parseParams['start_post_trialCode'] = 18;
    parseParams['start_pre_trialCode'] = 16;
    parseParams['start_spontCode'] = 35;
    parseParams['start_trialCode'] = 10;
    parseParams['start_wait_barCode'] = 23;
    parseParams['start_wait_fixationCode'] = 20;
    parseParams['startITICode'] = 12;
    parseParams['stim_numCode'] = 48;
    parseParams['stim_timeCode'] = 46;
    parseParams['stimColors'] = [];
    parseParams['stimdurCode'] = 51;
    parseParams['stimHeightCode'] = 59;
    parseParams['stimIDCode'] = 49;
    parseParams['stimIDOffset'] = 200;
    parseParams['stimOffCode'] = 39;
    parseParams['stimOnCode'] = 38;
    parseParams['stimRotOffset'] = 3736;
    parseParams['stimShapeCode'] = 60;
    parseParams['stimWidthCode'] = 58;
    parseParams['stop_spontCode'] = 36;
    parseParams['stop_trialCode'] = 11;
    parseParams['strobeBitChannel'] = 'AD17';
    parseParams['strobeThresh'] = 2;
    parseParams['targets_offCode'] = 41;
    parseParams['targets_onCode'] = 40;
    parseParams['test_offCode'] = 28;
    parseParams['test_onCode'] = 27;
    parseParams['UninitiatedTrialCode'] = 2;
    parseParams['unpauseCode'] = 101;
    parseParams['USER_ABORTCode'] = 1;
    parseParams['WRONG_RESPCode'] = 7;
    parseParams['yOffset'] = 600;
    return parseParams

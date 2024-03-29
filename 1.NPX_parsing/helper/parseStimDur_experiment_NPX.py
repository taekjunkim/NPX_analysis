#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 18:33:18 2019

parse nex file using nexfile provided by "https://www.neuroexplorer.com/downloadspage/"
neo.io.NeuroExplorerIO has some errors
1) cannot load waveform
2) cannot read onset time of each analog signal fragment

To filter out spikes for a constant period of time regardless of the stimulus duration,
spikes are selected within the time windows ('pdON - preTime' to 'pdON + 0.8'). 

This is different from parseTJexperiment_NPX.py where the time windows are set by 
('pdON - preTime' to 'pdOFF + postTime')

@author: taekjunkim
"""
#%%
# readSGLX.py is useful to understand how to handle raw data
import numpy as np;
import os.path
import pandas as pd; 

#%%
def main(bin_filename, dat_filename, prevTime, numConds, imec_filename, app):

    ### spikets
    """
    neuronIDs and spike_timing from the sorted output
    """
    task_index_in_combine = int(app.tasknum_lineEdit.text()); 
    man_sorted = app.sorted_checkbox.isChecked(); 

    id_all, spikets_all, chpos_all = get_spikeTS(imec_filename, task_index_in_combine, man_sorted); 

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

    markerts, pdOnTS, pdOffTS = get_event_ts(bin_filename, markervals_str); 

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

    counter = 0; 
    ### points where individual trials begin
    stimITIOns = np.where(markervals==parseParams['startITICode'])[0];     
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
                                            (spikets[j] <= (stimStructs[sIndex]['pdOn'][inst]+0.8)))[0];
                else:
                    spikeIndices = np.where((spikets[j] >= (stimOnTime-prevTime)) & 
                                            (spikets[j] <= (stimOnTime+0.8)))[0];
                                            
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
def get_event_ts(bin_filename, markervals_str):
   
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
    syncCh = int(metaDict['syncNiChan']); 
    syncONs = np.where(rawData[syncCh,:niSampRate*20]
                      >np.max(rawData[syncCh,:niSampRate*20])*0.5)[0];    
    for p in range(10):
        if syncONs[p+1]-syncONs[p]==1:
            syncON = syncONs[p]; 
            break; 

    ### read digit signal
    digitCh = np.shape(rawData)[0]-1;   # the last channel     
    digit_signal = rawData[digitCh, :]; 
    digit_diff = digit_signal[1:] - digit_signal[:-1]; 

    ### time (ms) with respect to syncON
    markerts = (np.where(digit_diff==2)[0] + 1 - syncON)/niSampRate; 
    pdOnTS_raw = (np.where(digit_diff==1)[0] + 1 - syncON)/niSampRate; 
    pdOffTS_raw = (np.where(digit_diff==-1)[0] + 1 - syncON)/niSampRate; 

    ### this is for photodiode generating pdOn for every frame
    pdOn_dist = pdOnTS_raw[1:] - pdOnTS_raw[:-1];
    pdOnTS = np.append(pdOnTS_raw[0],
                       pdOnTS_raw[np.where(pdOn_dist>0.02)[0]+1]);

    pdOff_dist = pdOffTS_raw[1:] - pdOffTS_raw[:-1];
    pdOffTS = np.append(pdOffTS_raw[0],
                        pdOffTS_raw[np.where(pdOff_dist>0.02)[0]+1]);

    """
    ### this is for photodiode generating continuous pdOn
    pdOnTS = pdOnTS_raw; 
    pdOffTS = pdOffTS_raw; 
    """
    
    if len(markerts)==len(markervals_str):
        print('Good: number of events are well matched');         
        pass; 
    else:
        markerts = np.zeros(len(markervals_str)); 
        stim_ons = np.where(markervals_str=='sample_on')[0]; 
        if len(stim_ons) == len(pdOnTS):
            print('number of pdONs is matched with "sample_on"');             
            markerts[stim_ons] = pdOnTS - 0.001;   # 1 ms earlier than pdOnTS  
        else:
            print('number of pdONs is not matched with "sample_on"'); 
            markerts[stim_ons] = pdOnTS[:len(stim_ons)] - 0.001; 

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
def get_spikeTS(imec_filename, task_index_in_combine, man_sorted):    

    imec_dataFolder = imec_filename[:(imec_filename.rfind('/')+1)]; 

    if os.path.exists(imec_dataFolder+'info/imec_datainfo.npy'):
        imec_info = np.load(imec_dataFolder+'info/imec_datainfo.npy', allow_pickle=True).item(); 
        imSampRate = imec_info['ap: imSampRate'][task_index_in_combine]; 

        rawdata = []; 
        binname = []; 
        syncON = imec_info['ap: firstSamp'][task_index_in_combine] + imec_info['ap: syncON'][task_index_in_combine]; 
    else:
        binname = imec_filename; 
        metaname = imec_filename[:-3]+'meta'; 

        metaDict = get_metaDict(metaname);
        imSampRate = float(metaDict['imSampRate']);
        nChan = int(metaDict['nSavedChans']);
        nFileSamp = int(int(metaDict['fileSizeBytes'])/(2*nChan));

        """
        with open(binname,'r') as fid:
            ### to get synON, read the first 10 sec signal
            rawdata = np.fromfile(fid,np.int16,count=int(imSampRate)*10*nChan).reshape([nChan,10*int(imSampRate)],order='F');
        """
        rawdata = np.memmap(binname, dtype='int16', mode='r', shape=(nChan,nFileSamp),order='F');

        syncONs = np.where(rawdata[384,:int(imSampRate*10)]>np.max(rawdata[384,:int(imSampRate*10)])*0.5)[0];
        for p in range(10):
            if syncONs[p+1]-syncONs[p]==1:
                syncON = syncONs[p]; 
                break; 

    st = np.load(imec_dataFolder+'spike_times.npy');
    sc = np.load(imec_dataFolder+'spike_clusters.npy');
    
    st = (st.astype(dtype='int64')-syncON)/imSampRate;


    if man_sorted==1:
        df = pd.read_csv(imec_dataFolder+'cluster_info.tsv', sep='\t');
        ch_map = np.load(imec_dataFolder+'channel_map.npy');  
        ch_pos = np.load(imec_dataFolder+'channel_positions.npy');  
        for i in np.arange(df.shape[0]):
            id_chmap = np.where(ch_map==df.loc[i,'ch'])[0]; 
            df.loc[i,'xc'] = ch_pos[id_chmap,0]; 
            df.loc[i,'yc'] = ch_pos[id_chmap,1]; 
    else:
        df = pd.read_csv(imec_dataFolder+'cluster_KSLabel.tsv', sep='\t');
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

        for i in np.arange(df.shape[0]):
            spk_now = np.where(sc==df.loc[i,'cluster_id'])[0]; 
            depths_now = np.nanmean(spikeDepths[spk_now]); 
            if ~np.isnan(depths_now):
                depths_now = round(depths_now); 

            df.loc[i,'xc'] = np.nan; 
            df.loc[i,'yc'] = depths_now; 

    
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
    
    del rawdata, binname;

    # timestamps in seconds    
    return id_all, spikets_all, chpos_all; 



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

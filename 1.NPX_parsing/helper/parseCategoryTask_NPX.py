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

#%%
def main(bin_filename, dat_filename, prevTime, maxNumTrials, imec_filename, app):

    ### spikets
    """
    neuronIDs and spike_timing from the sorted output
    """
    task_index_in_combine = int(app.tasknum_lineEdit.text()); 
    man_sorted = app.sorted_checkbox.isChecked(); 

    if imec_filename==[]:
        spikets = [np.array([0,1,2,3])]; # fake spikets: 1 unit, 4 spikes
        numNeurons = len(spikets); 
        neuronid = np.array([0]); 
    else:
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
    experiment['bin_filename'] = bin_filename;    ## changed for NPX
    experiment['dat_filename'] = dat_filename;   ## changed for NPX  
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
    experiment['numTrials'] = 0; 
    experiment['refStims'] = [];       
    experiment['rots'] = []; 
    experiment['cats'] = []; 
    experiment['colors'] = [];       

    rots = np.arange(0,360,45); # the standard rotations
    numNormalColors = 5; # stim color, distracor color, distGray, bg_before and bg_during

    # Prepare StimStructs
    stimStructs = []; 
    maxNumTrials = 3500; 
    for i in np.arange(maxNumTrials):
        stimStructs.append(dict());
        stimStructs[i]['timeOn'] = [];
        stimStructs[i]['timeOff'] = [];

        # stim params
        stimStructs[i]['stimID'] = 0; 
        stimStructs[i]['rotation'] = 0; 
        stimStructs[i]['category'] = 0; 
        stimStructs[i]['stimColor'] = 0; 

        # distractor params
        stimStructs[i]['distID'] = 0; 
        stimStructs[i]['distN'] = 0; 
        stimStructs[i]['distD'] = 0; 
        stimStructs[i]['distS'] = 0.0; 
        stimStructs[i]['distM'] = 0;   # modeID
        stimStructs[i]['varyID'] = 0;   # varyID        

        stimStructs[i]['reaction_time'] = []; 
        stimStructs[i]['trialCode'] = 0; 
        stimStructs[i]['breakfixTime'] = 0; 
        stimStructs[i]['pdOn'] = [];
        stimStructs[i]['pdOff'] = [];

        stimStructs[i]['neurons'] = [];        
        for j in np.arange(numNeurons):
            stimStructs[i]['neurons'].append(dict());

    ### get trial independent codes (taks parameters)
    counter = 0; 
    if markervals[counter] != parseParams['mon_ppdCode']:
        print('markerval #'+str(counter)+' was not mon_ppdCode');
    else:
        experiment['monppd'] = markervals[counter+1];
        counter = counter + 2;

    if markervals[counter] != parseParams['fixXCode']:
        print('markerval #'+str(counter)+' was not fixXCode');
    else:
        experiment['fixX'] = markervals[counter+1];
        counter = counter + 2;

    if markervals[counter] != parseParams['fixYCode']:
        print('markerval #'+str(counter)+' was not fixYCode');
    else:
        experiment['fixY'] = markervals[counter+1];
        counter = counter + 2;

    if markervals[counter] != parseParams['rfxCode']:
        print('markerval #'+str(counter)+' was not rfxCode');
    else:
        experiment['rfx'] = markervals[counter+1] - parseParams['plexYOffsetCode'];
        counter = counter + 2;
        
    if markervals[counter] != parseParams['rfyCode']:
        print('markerval #'+str(counter)+' was not rfyCode');
    else:
        experiment['rfy'] = markervals[counter+1] - parseParams['plexYOffsetCode'];
        counter = counter + 2;
    
    if markervals[counter] != parseParams['itiCode']:
        print('markerval #'+str(counter)+' was not itiCode');
    else:
        experiment['iti'] = markervals[counter+1];
        counter = counter + 2;

    if markervals[counter] != parseParams['isiCode']:
        print('markerval #'+str(counter)+' was not isiCode');
    else:
        experiment['isi'] = markervals[counter+1];
        counter = counter + 2;
        
    if markervals[counter] != parseParams['stim_timeCode']:
        print('markerval #'+str(counter)+' was not stim_timeCode');
    else:
        experiment['StimDur'] = markervals[counter+1]; 
        counter = counter + 2;

    ### next three numbers are rDistractor, rSpace and flankSize
    experiment['flankSize'] = markervals[counter]; 
    experiment['distShape'] = markervals[counter+1]; 

    counter = counter+2; 

    ### parse stim IDs
    numCounter = 0; 
    numCounter = markervals[counter] - parseParams['stimIDOffset']; 
    counter = counter+1; 

    if numCounter != 0: # the code for all the stims
        for i in np.arange(numCounter):
            stimNow = markervals[counter] - parseParams['stimIDOffset']; 
            experiment['refStims'].append(stimNow); 
            counter = counter + 1; 

    ### encoding stimulus rotations
    numCounter = 0; 
    numCounter = markervals[counter] - parseParams['stimRotOffset']; 
    counter = counter + 1; 

    if numCounter != 0: # the code for all the rots
        for i in np.arange(numCounter):
            rotNow = markervals[counter] - parseParams['stimRotOffset']; 
            experiment['rots'].append(rotNow); 
            counter = counter + 1; 

    ### encoding stimulus categories
    numCounter = 0; 
    numCounter = markervals[counter] - parseParams['stimRotOffset']; 
    counter = counter + 1; 

    if numCounter != 0: # the code for all the categories
        for i in np.arange(numCounter):
            catNow = markervals[counter] - parseParams['stimRotOffset']; 
            experiment['cats'].append(catNow); 
            counter = counter + 1; 

    ### color code
    if markervals[counter] != parseParams['colorCode']:
        print(f'error: markerval #{counter} was not colorCode'); 
    else:
        counter = counter + 1; 

    colors = np.where(markervals[counter:counter + 3*numNormalColors] >= parseParams['stimRotOffset'])[0]; 
    if (len(colors)==0) or (len(colors)!=3*numNormalColors):
        print(f'error: markervals# {counter}:{counter+3*numNormalColors} were not colors'); 
    else:
        tempColors = markervals[counter+colors]; 
        for j in np.arange(numNormalColors):
            colorNow = tempColors[j*3:(j+1)*3] - parseParams['stimRotOffset']; 
            experiment['colors'].append(colorNow); 
        counter = counter + len(colors); 

    # Prepare to get stimulus information parameters                
    # Find the startITI codes                
    stimITIOns = np.where(markervals==parseParams['startITICode'])[0];
    if stimITIOns[0] != counter:
        print('The first start_iti code is offset');
    stimOns = np.where(markervals==parseParams['stimOnCode'])[0]; # created but not used       
    
    error_indices = []; 
    completedITIs = 0; 

    # Outer loop has behavioral exception handling
    # all stim codes in inner loop
    for i in np.arange(len(stimITIOns)): # the file should end with 
                                           # a startITI that we don't care about    
        if stimITIOns[i] < counter:
            continue; 

        index = stimITIOns[i] + 2;  # eye start and endITI
        if len(markervals) < index+2: # if there are too few codes, its the last startITI
            break; 

        next_code = markervals[index]; # either endITI/pause or wrong code

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

        next_code2 = markervals[index+2]; # this is fix acquired after fix_On 
        if next_code2 == parseParams['fixAcquiredCode']:
            pass;
        elif next_code2 == parseParams['UninitiatedTrialCode']:
            if markervals[index+3] != parseParams['fix_offCode']:
                error_indices.append(index+1);
                print('Found non start_iti code '+str(markervals[index+2])+
                      ' after Uninitiated trial at '+str(index+2));
            continue;
        else:
            print('Found bad code '+str(next_code2)+' after end_iti at index '
                  +str(stimITIOns[i]+2));
            error_indices.append(index+1);           
            continue;

        ndex = index + 3;  # this is index where we're at if fixAcquired
                           # this would be stimIdcode followed by stimulus number
        trialCode = []; 

        ### inner loop (through all stimulus codes in a given trial)
        while (trialCode == []):        

            optionalCode = 0;
            stimCode = markervals[ndex+optionalCode];
            
            if stimCode == parseParams['fixLost']:
                if hasValidBreakFixMultiMatching(ndex+optionalCode,markervals,parseParams):
                    trialCode = parseParams['breakFixCode'];
                    continue;
            elif stimCode != parseParams['stimIDCode']:
                print('Found '+str(stimCode)+' as a stimID or breakfix code at stim time '
                      +str(markerts[ndex+optionalCode])+' at index '+str(ndex+optionalCode));
                print('continuing from next start_iti');                            
                error_indices.append(ndex+optionalCode);
                trialCode = parseParams['codeError'];
                continue;

            # stimID    
            if markervals[ndex+1+optionalCode] == parseParams['fixLost']:
                if hasValidBreakFixMultiMatching(ndex+1+optionalCode,markervals,parseParams):
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

            # colorCode
            if ((markervals[ndex+2+optionalCode] >= parseParams['stimIDOffset']) 
                  and (markervals[ndex+2+optionalCode] < parseParams['stimRotOffset'])): 
                colorCodeToStore = markervals[ndex+2+optionalCode];
            else:
                print('Found '+str(markervals[ndex+2])+' as a color code at stim time '
                      +str(markerts[ndex+2+optionalCode])+' at index '+str(ndex+2+optionalCode));
                print('continuing from next start_iti');                            
                error_indices.append(ndex+optionalCode+2);
                trialCode = parseParams['codeError'];
                continue;                

            # rotCode
            codeIndex = ndex + 3 + optionalCode
            rotCode = markervals[codeIndex]; 
            if rotCode == parseParams['fixLost']:
                if hasValidBreakFixMultiMatching(codeIndex,markervals,parseParams):
                    trialCode = parseParams['breakFixCode'];
                    continue;
            elif rotCode != parseParams['rotIDCode']:          
                print('Missing rotID or fixlost code, found '+str(rotCode)+' at '+str(codeIndex))
                print('continuing from next start_iti');                                        
                error_indices.append(codeIndex);
                trialCode = parseParams['codeError'];
                continue;                

            codeIndex = ndex + 4 + optionalCode
            rotCode2 = markervals[codeIndex]; 
            if rotCode2 == parseParams['fixLost']:
                if hasValidBreakFixMultiMatching(codeIndex,markervals,parseParams):
                    trialCode = parseParams['breakFixCode'];
                    continue;
            elif ((rotCode2 >= parseParams['stimRotOffset']) 
                and (rotCode2 <= parseParams['stimRotOffset'])+360):  ## success!
                rotCodeToStore = rotCode2; 
            else:
                print('Missing rotID or fixlost code, found '+str(rotCode2)+' at '+str(codeIndex))
                print('continuing from next start_iti');                                        
                error_indices.append(codeIndex);
                trialCode = parseParams['codeError'];
                continue;                

            # catCode 
            codeIndex = ndex + 5 + optionalCode; 
            catCode = markervals[codeIndex]; 
            if catCode == parseParams['fixLost']:
                if hasValidBreakFixMultiMatching(codeIndex,markervals,parseParams):
                    trialCode = parseParams['breakFixCode'];
                    continue;
            elif catCode != parseParams['rotIDCode']: # using rotIDCode for category code
                print('Missing Category code, found '+str(rotCode)+' at '+str(codeIndex))
                print('continuing from next start_iti');                                        
                error_indices.append(codeIndex);
                trialCode = parseParams['codeError'];
                continue;                

            codeIndex = ndex + 6 + optionalCode; 
            catCode2 = markervals[codeIndex]; 
            if catCode2 == parseParams['fixLost']:
                if hasValidBreakFixMultiMatching(codeIndex,markervals,parseParams):
                    trialCode = parseParams['breakFixCode'];
                    continue;
            elif ((catCode2 >= parseParams['stimRotOffset']) 
                and (catCode2 <= parseParams['stimRotOffset'])+360):  ## success!
                catCodeToStore = catCode2; 
            else:
                print('Missing Category code, found '+str(rotCode2)+' at '+str(codeIndex))
                print('continuing from next start_iti');                                        
                error_indices.append(codeIndex);
                trialCode = parseParams['codeError'];
                continue;          

            # distCode 
            codeIndex = ndex + 7 + optionalCode; 
            distCode = markervals[codeIndex]; 
            if distCode == parseParams['fixLost']:
                if hasValidBreakFixMultiMatching(codeIndex,markervals,parseParams):
                    trialCode = parseParams['breakFixCode']; 
                    continue;
            elif distCode != parseParams['occl_infoCode']: # using occl_infoCode for dist code
                print('Missing Dist code, found '+str(rotCode)+' at '+str(codeIndex)); 
                print('continuing from next start_iti');                                        
                error_indices.append(codeIndex); 
                trialCode = parseParams['codeError']; 
                continue;                

            # next code is either fixlost or distID
            codeIndex = ndex + 8 + optionalCode; 
            distCode2 = markervals[codeIndex]; 
            if distCode2 == parseParams['fixLost']:
                if hasValidBreakFixMultiMatching(codeIndex,markervals,parseParams):
                    trialCode = parseParams['breakFixCode'];
                    continue;
            elif ((distCode2 >= parseParams['stimRotOffset']) 
                and (distCode2 <= parseParams['stimRotOffset'])+360):  ## success!
                distCodeToStore = distCode2; 
            else:
                print('Missing Dist code, found '+str(rotCode2)+' at '+str(codeIndex))
                print('continuing from next start_iti');                                        
                error_indices.append(codeIndex);
                trialCode = parseParams['codeError'];
                continue;          

            ## next code is either fixlost or distractor number
            codeIndex = ndex + 9 + optionalCode;
            distNCode = markervals[codeIndex]; 
            if distNCode == parseParams['fixLost']:
                if hasValidBreakFixMultiMatching(codeIndex,markervals,parseParams):
                    trialCode = parseParams['breakFixCode'];
                    continue;
            elif ((distNCode >= parseParams['stimRotOffset']) 
                and (distNCode <= parseParams['stimRotOffset'])+360):  ## success!
                distNCodeToStore = distNCode;
            else: 
                print('Missing DistN code, found '+str(distNCode)+' at '+str(codeIndex))
                print('continuing from next start_iti');                                        
                error_indices.append(codeIndex); 
                trialCode = parseParams['codeError'];
                continue;          

            ## next code is either fixlost or distractor distance
            codeIndex = ndex + 10 + optionalCode;
            distDCode = markervals[codeIndex]; 
            if distDCode == parseParams['fixLost']:
                if hasValidBreakFixMultiMatching(codeIndex,markervals,parseParams):
                    trialCode = parseParams['breakFixCode'];
                    continue;
            elif ((distDCode >= parseParams['stimRotOffset'])
                and (distDCode <= parseParams['stimRotOffset'])+360):  ## success!
                distDCodeToStore = distDCode;
            else: 
                print('Missing DistN code, found '+str(distDCode)+' at '+str(codeIndex))
                print('continuing from next start_iti');                                        
                error_indices.append(codeIndex); 
                trialCode = parseParams['codeError'];
                continue;         

            ## next code is either fixlost or distractor mode code which is either 1 or 2
            codeIndex = ndex + 11 + optionalCode;
            distMCode = markervals[codeIndex];
            if distMCode == parseParams['fixLost']:
                if hasValidBreakFixMultiMatching(codeIndex,markervals,parseParams):
                    trialCode = parseParams['breakFixCode'];
                    continue;
            elif ((distMCode >= parseParams['stimRotOffset'])
                and (distMCode <= parseParams['stimRotOffset'])+2):  ## either 1 or 2!
                distMCodeToStore = distMCode;
            else: 
                print('Missing distM code, found '+str(distMCode)+' at '+str(codeIndex))
                print('continuing from next start_iti');                                        
                error_indices.append(codeIndex); 
                trialCode = parseParams['codeError'];
                continue;          

            ## next code is either fixlost or distractor size fraction
            codeIndex = ndex + 12 + optionalCode; 
            distSCode = markervals[codeIndex]; 
            if distSCode == parseParams['fixLost']:
                if hasValidBreakFixMultiMatching(codeIndex,markervals,parseParams):
                    trialCode = parseParams['breakFixCode'];
                    continue;
            elif ((distSCode >= parseParams['stimRotOffset']) 
                and (distSCode <= parseParams['stimRotOffset'])+360):  ## success!
                distSCodeToStore = distSCode;
            else: 
                print('Missing DistS code, found '+str(distSCode)+' at '+str(codeIndex))
                print('continuing from next start_iti');                                        
                error_indices.append(codeIndex); 
                trialCode = parseParams['codeError'];
                continue;                         

            ## next code is either fixlost or varyID
            codeIndex = ndex + 13 + optionalCode; 
            varyIDCode = markervals[codeIndex]; 
            if varyIDCode == parseParams['fixLost']:
                if hasValidBreakFixMultiMatching(codeIndex,markervals,parseParams):
                    trialCode = parseParams['breakFixCode'];
                    continue;
            elif ((varyIDCode >= parseParams['stimRotOffset']) 
                and (varyIDCode <= parseParams['stimRotOffset'])+360):  ## success!
                varyIDCodeToStore = varyIDCode;
            else: 
                print('Missing varyID code, found '+str(varyIDCode)+' at '+str(codeIndex))
                print('continuing from next start_iti');                                        
                error_indices.append(codeIndex); 
                trialCode = parseParams['codeError'];
                continue;                                         

            ## next code is either fixlost or stimOn
            codeIndex = ndex + 14 + optionalCode;
            code = markervals[codeIndex]; 
            if code == parseParams['fixLost']:
                if hasValidBreakFixMultiMatching(codeIndex,markervals,parseParams):
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
                
            ## Having made it here we can now call this a completed stimulus presentation and record the results
            sIndex = stimIDCodeToStore - parseParams['stimIDOffset'];
            rotIndex = np.where(rots == (rotCodeToStore - parseParams['stimRotOffset']))[0]; 
            colorIndex = colorCodeToStore - parseParams['stimIDOffset']; 

            if len(rotIndex) == 0:
                print(f'Error: did not find correct rotation code following StimON, at StimCode: {stimCode}. Continue ITI {i}')           
                rotIndex = 1; 

            stimStructs[completedITIs-1]['timeOn'] = stimOnTime; 
            stimStructs[completedITIs-1]['rotation'] = rotCodeToStore - parseParams['stimRotOffset']; 
            stimStructs[completedITIs-1]['category'] = catCodeToStore - parseParams['stimRotOffset']; 
            stimStructs[completedITIs-1]['stimID'] = sIndex; 
            stimStructs[completedITIs-1]['stimColor'] = colorIndex;

            # distractor parameters
            stimStructs[completedITIs-1]['distID'] = distCodeToStore - parseParams['stimRotOffset']; 
            stimStructs[completedITIs-1]['distN'] = distNCodeToStore - parseParams['stimRotOffset']; 
            stimStructs[completedITIs-1]['distD'] = distDCodeToStore - parseParams['stimRotOffset']; 
            stimStructs[completedITIs-1]['distS'] = (distSCodeToStore - parseParams['stimRotOffset'])/100.0; 
            stimStructs[completedITIs-1]['distM'] = distMCodeToStore - parseParams['stimRotOffset'];  # modeID
            stimStructs[completedITIs-1]['varyID'] = varyIDCodeToStore - parseParams['stimRotOffset'];  # varyID            


            codeIndex = ndex + 15 + optionalCode; 

            ## next code is either fixlost or stimOff
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
                if (code == parseParams['stimOffCode']) and (markervals[codeIndex+1] == parseParams['NO_RESPCode']):
                    stimStructs[completedITIs-1]['breakfixTime'] = markerts[codeIndex] - markerts[codeIndex-1];
                    trialCode = parseParams['NO_RESPCode']; 
                    stimStructs[completedITIs-1]['trialCode'] = trialCode; 
                stimOffTime = markerts[codeIndex];
                stimStructs[completedITIs-1]['timeOff'] = stimOffTime;
                rxtTime = markervals[codeIndex+1];  
                stimStructs[completedITIs-1]['reaction_time'] = rxtTime - parseParams['stimIDOffset']; 

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
                    stimStructs[completedITIs-1]['pdOn'].append(pdOnTS[pdOnsAfter[0]]); 
                    stimStructs[completedITIs-1]['pdOff'].append(pdOffTS[pdOffsAfter[0]]);                    

            # find which happens first stimOnTime or pdOn, start recording
            # neural data at this time - prevTime
            if len(stimStructs[completedITIs-1]['pdOn'])>0:
                #potentialStart = np.min([stimOnTime, stimStructs[completedITIs-1]['pdOn'][0]]); 
                potentialStart = stimStructs[completedITIs-1]['pdOn'][0]; 
            else:
                potentialStart = stimOnTime; 

            # find which happens last stimOffTime or pdOff, stop recording
            # neural data at this time + postTime
            if len(stimStructs[completedITIs-1]['pdOff'])>0:
                #potentialEnd = np.max([stimOffTime, stimStructs[completedITIs-1]['pdOff'][0]]); 
                potentialEnd = np.max([potentialStart+300, stimStructs[completedITIs-1]['pdOff'][0]]); 
            else:
                potentialEnd = stimOffTime; 

    
            ## now get neural data: single trial gives one spike train
            ## in other tasks, single trial can give multiple spike trains associated with multiple conditions
            for j in np.arange(numNeurons):
                mySpikes = np.array([]);

                spikeIndices = np.where((spikets[j] >= (potentialStart-prevTime)) & 
                                        (spikets[j] <= (potentialEnd+postTime)))[0];
                                            
                if len(spikeIndices)>0:
                    mySpikes = np.append(mySpikes,spikets[j][spikeIndices]);
                stimStructs[completedITIs-1]['neurons'][j]['spikes'] = mySpikes; 


            ## next code is either EARLY_RELEASECode, correctCode, WRONG_RESPCode or fixLost
            codeIndex = ndex + 17 + optionalCode;
            code = markervals[codeIndex]; 

            if code == parseParams['EARLY_RELEASECode']:
                trialCode = parseParams['EARLY_RELEASECode']; 
                continue; 
            elif code == parseParams['correctCode']: 
                trialCode = parseParams['correctCode']; 
                stimStructs[completedITIs-1]['trialCode'] = 1; 
                continue;
            elif code == parseParams['WRONG_RESPCode']: 
                trialCode = parseParams['WRONG_RESPCode']; 
                stimStructs[completedITIs-1]['trialCode'] = 2; 
                continue;
            elif code == parseParams['fixLost']:
                if hasValidBreakFixMultiMatching(codeIndex,markervals,parseParams):
                    trialCode = parseParams['breakFixCode'];
                    continue;

#%% add stimStructs to experiment output, then return
    experiment['stimStructs'] = stimStructs;                        
    experiment['errors'] = error_indices; 
    experiment['numTrials'] = completedITIs;    
    
    return experiment;

#%% hasValidBreakFixMultiMatching
def hasValidBreakFixMultiMatching(ndex, markervals, parseParams):
    if markervals[ndex] != parseParams['fixLost']:
        print('missing breakFixCode after '+str(markervals[ndex])+
              ' at index '+str(ndex));
    if markervals[ndex+1] != parseParams['fix_offCode']:
        print('missing fix_offCode after '+str(markervals[ndex+1])+
              ' at index '+str(ndex));
    if markervals[ndex+2] != parseParams['eye_stopCode']:
        print('missing eye_stopCode after '+str(markervals[ndex+2])+
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
    pdOnTS = (np.where(digit_diff==1)[0] + 1 - syncON)/niSampRate; 
    pdOffTS = (np.where(digit_diff==-1)[0] + 1 - syncON)/niSampRate; 

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

        if 'ap: syncOFF' in imec_info.keys():
            syncDur_nidq = imec_info['nidq: syncOFF'][task_index_in_combine] - imec_info['nidq: syncON'][task_index_in_combine]; 
            syncDur_nidq_sec = syncDur_nidq / imec_info['nidq: SampRate'][task_index_in_combine]; 

            # adjusted imSampRate
            syncDur_ap = imec_info['ap: syncOFF'][task_index_in_combine] - imec_info['ap: syncON'][task_index_in_combine]; 
            imSampRate = syncDur_ap / syncDur_nidq_sec; 

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
            if syncONs[p+10]-syncONs[p]==10:
                syncON = syncONs[p]; 
                break; 
    
        syncOFFs = np.where(
            rawdata[384,-int(imSampRate*10):] > np.max(rawdata[384, -int(imSampRate*10):])*0.5
        )[0] + nFileSamp - imSampRate*10;    
        for p in range(10):
            if syncOFFs[-p-1]-syncOFFs[-p-11]==10:
                syncOFF = syncOFFs[-p-1]; 
                break; 
        
        syncDur_ap = syncOFF - syncON; 

        ### check NIDQ
        idx_slash = []; 
        for c in np.arange(len(binname)):
            if binname[c]=='/':
                idx_slash.append(c); 

        nidq_bin = glob.glob(binname[:idx_slash[-2]]+'/*nidq.bin')[0]
        nidq_metaname = nidq_bin[:-3]+'meta'; 
        nidq_meta = get_metaDict(nidq_metaname); 
        nidq_syncCH = int(nidq_meta['syncNiChan'])
        nidq_nChan = int(nidq_meta['nSavedChans']); 
        nidq_nFileSamp = int(int(nidq_meta['fileSizeBytes'])/(2*nidq_nChan)); 
        nidq_SampRate = int(nidq_meta['niSampRate']);         
        
        nidq_data = np.memmap(nidq_bin, dtype='int16', 
                              shape=(nidq_nFileSamp, nidq_nChan), offset=0, order='C'); 
        nidq_syncONs = np.where(
            nidq_data[:nidq_SampRate*20, nidq_syncCH]
            > np.max(nidq_data[:nidq_SampRate*20, nidq_syncCH])*0.5
        )[0];    
        nidq_syncOFFs = np.where(
            nidq_data[-nidq_SampRate*20:, nidq_syncCH]
            > np.max(nidq_data[-nidq_SampRate*20:, nidq_syncCH])*0.5
        )[0] + nidq_nFileSamp - nidq_SampRate*20;    
        for p in range(10):
            if nidq_syncONs[p+10]-nidq_syncONs[p]==10:
                nidq_syncON = nidq_syncONs[p]; 
                break; 
        for p in range(10):
            if nidq_syncOFFs[-p-1]-nidq_syncOFFs[-p-11]==10:
                nidq_syncOFF = nidq_syncOFFs[-p-1]; 
                break; 
        syncDur_nidq = nidq_syncOFF - nidq_syncON; 
        syncDur_nidq_sec = syncDur_nidq / nidq_SampRate; 

        ### adjust imSampRate    
        imSampRate = syncDur_ap / syncDur_nidq_sec; 

 
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
    parseParams['fixXCode'] = 76;
    parseParams['fixYCode'] = 77;        
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
    parseParams['mon_ppdCode'] = 78;    
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

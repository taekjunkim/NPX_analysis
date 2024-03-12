#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 7 2022

CategoryTask_NPX_Analysis.py

@author: taekjunkim
"""

#%% 
import matplotlib.pyplot as plt;
import scipy.optimize as opt;
import glob
import numpy as np; 
import os; 
#import sys;
#sys.path.append('./helper'); 
#sys.path.append('./makeSDF'); 
import makeSDF;
import parseCategoryTask_NPX as parseCategory_NPX;
import json;
import gzip;


#%%
def main(app):
    #%% This needs to be defined
    dat_filename = app.dat_file; 
    imec_filename = app.imec_file; 
    task_folder = dat_filename[:(dat_filename.rfind('/')+1)]; 
    bin_filename = glob.glob(task_folder+'*.nidq.bin')[0]; 

    prevTime = 0.3; 
    markervals, markervals_str = parseCategory_NPX.get_markervals(dat_filename); 

    maxNumTrials = 3500; 
    experiment = parseCategory_NPX.main(bin_filename, dat_filename, prevTime, maxNumTrials, imec_filename, app); 

    n_refStim = len(experiment['refStims']); 
    numTrials = experiment['numTrials']; 
    stimStructs = experiment['stimStructs']; 

    ### Behavior Matrix ###
    behVec = np.empty((numTrials,11));  
    # columns: stim_index, category, distN, distM, trialCode, RT, distD, distS, distID/distColor, stimColor, varyID
    behVec[:] = np.nan; 

    # construct a vector with behavioral metrics for each trial:
    # columns are as follows: [stimindex category nDistractors mode correctORerror RT] 
    # col0: stim index
    # col1: category
        # 1 (4-pronged), 2 (3-pronged)
    # col2: number of distractors
    # col3: mode of distractors
        # 1 (all same), 2 (random)    
    # col4: trialCode        
        # 1 (correct), 2 (wrong)        
    # col5: reaction time
    # col6: targ-dist distance (pix)
    # col7: size of distractor (fraction of flankSize)
    # col8: color of distractor
        # 0 (dist color), 1 (targ color), 2 (gray)
        # 3 (dist color CC), 4 (targ color CC), 5 (gray CC)
    # col9: color of target
        # 1 (targ color), 2 (dist color)
    # col10: vary condition number
        # 1 (NumVary), 2 (DistVary), 3 (ColorVary)        
        # 4 (SizeVary), 5 (UncrowdVary), 6 (NumSizeColorVary)                                                                     

    for i in range(numTrials):
        if stimStructs[i]['stimID']>0:
            stim_idx =  np.where((experiment['refStims']==stimStructs[i]['stimID']) & 
                                (experiment['rots']==stimStructs[i]['rotation']))[0]; 
            behVec[i,0] = stim_idx;                              # stim index
            behVec[i,1] = stimStructs[i]['category'];            # category
                                                                # 1 (4-pronged), 2 (3-pronged)
            behVec[i,2] = stimStructs[i]['distN'];               # number of distractors
            behVec[i,3] = stimStructs[i]['distM'];               # mode of distractors
                                                                # 1 (all same), 2 (random)        
            behVec[i,4] = stimStructs[i]['trialCode'];           # 1 (correct), 2 (wrong)
            if stimStructs[i]['trialCode']<3:  # if its a correct or error trial measure RT
                behVec[i,5] = stimStructs[i]['reaction_time'];   # reaction time                                
            behVec[i,6] = stimStructs[i]['distD'];               # target-distractor distance (pix)
            behVec[i,7] = stimStructs[i]['distS'];               # size of distractor (fraction of flankSize)
            behVec[i,8] = stimStructs[i]['distID'];              # color of distractor
                                                                # 0 (dist color), 1 (targ color), 2 (gray)
                                                                # 3 (dist color CC)
                                                                # 4 (targ color CC)
                                                                # 5 (gray CC)
            behVec[i,9] = stimStructs[i]['stimColor'];           # color of target
                                                                # 1 (targ color), 2 (dist color)
            behVec[i,10] = stimStructs[i]['varyID'];             # vary condition number 
                                                                # 1 (NumVary), 2 (DistVary), 3 (ColorVary)        
                                                             # 4 (SizeVary), 5 (UncrowdVary), 6 (NumSizeColorVary)                                                                     

    pCorrect = dict(); 
    correct_RT = dict(); 
    wrong_RT = dict(); 

    ### No Distractor ###
    # col0: stim index
    # col1: category
        # 1 (4-pronged), 2 (3-pronged)
    # col2: number of distractors
    # col3: mode of distractors
        # 1 (all same), 2 (random)    
    # col4: trialCode        
        # 1 (correct), 2 (wrong)        
    # col5: reaction time
    # col6: targ-dist distance (pix)
    # col7: size of distractor (fraction of flankSize)
    # col8: color of distractor
        # 0 (dist color), 1 (targ color), 2 (gray)
        # 3 (dist color CC), 4 (targ color CC), 5 (gray CC)
    # col9: color of target
        # 1 (targ color), 2 (dist color)
    # col10: vary condition number
        # 1 (NumVary), 2 (DistVary), 3 (ColorVary)        
        # 4 (SizeVary), 5 (UncrowdVary), 6 (NumSizeColorVary)       

    pCorrect['noDist'] = np.full([1,n_refStim], np.nan); 
    correct_RT['noDist'] = np.full([1,n_refStim], np.nan); 
    wrong_RT['noDist'] = np.full([1,n_refStim], np.nan); 
    xlabels = []; 
    ## plot performance & RT by stimID
    for i in np.arange(n_refStim):
        all_thisID = np.where((behVec[:,0]==i) & (behVec[:,2]==0))[0]; 
        correct_thisID = np.where((behVec[:,0]==i) &
                                  (behVec[:,2]==0) &
                                  (behVec[:,4]==1))[0]; 
        wrong_thisID = np.where((behVec[:,0]==i) &
                                (behVec[:,2]==0) &        
                                (behVec[:,4]==2))[0];

        pCorrect['noDist'][0,i] = 100 * (len(correct_thisID) / len(all_thisID)); 
        correct_RT['noDist'][0,i] = np.nanmean(behVec[correct_thisID,5]); 
        wrong_RT['noDist'][0,i] = np.nanmean(behVec[wrong_thisID,5]); 
        xlabels.append(f'id{i}'); 

    plt.figure(); 
    ax1 = plt.subplot(2,2,1); 
    ax3 = plt.subplot(2,2,3); 
    bar_position = np.arange(1.5,1.5+n_refStim*3,3); 
    # to separate cat1 and cat2
    bar_position[int(n_refStim/2):] = bar_position[int(n_refStim/2):] + 2;  

    ax1.bar(bar_position, pCorrect['noDist'][0,:], facecolor=[0.5,0.5,0.5]);  
    ax1.plot([0, bar_position[int(n_refStim/2)-1]+1.5], 
             [np.nanmean(pCorrect['noDist'][0,:int(n_refStim/2)]), 
              np.nanmean(pCorrect['noDist'][0,:int(n_refStim/2)])],
              color='r', linewidth = 2); 
    ax1.plot([bar_position[int(n_refStim/2)]-1.5, bar_position[-1]+1.5], 
             [np.nanmean(pCorrect['noDist'][0,int(n_refStim/2):]), 
              np.nanmean(pCorrect['noDist'][0,int(n_refStim/2):])],
              color='r', linewidth = 2); 
    ax1.set_xlim([0, bar_position[-1]+1.5]); 
    ax1.set_ylim([0, 100]); 
    ax1.set_xticks(bar_position); 
    ax1.set_xticklabels(xlabels); 
    ax1.set_xlabel('Stimulus ID'); 
    ax1.set_ylabel('%Correct peformance'); 
    ax1.set_title('No distractor'); 

    bar_position_correct = bar_position - 0.5; 
    bar_position_wrong = bar_position + 0.5; 

    ax3.bar(bar_position_correct, correct_RT['noDist'][0,:],width=0.5, color='r', label='correct');
    ax3.bar(bar_position_wrong, wrong_RT['noDist'][0,:], width=0.5, color='b', label='wrong'); 
    ax3.plot(0,np.nanmean(correct_RT['noDist'][0,:int(n_refStim/2)]),'r>', markerfacecolor='r'); 
    ax3.plot(0,np.nanmean(wrong_RT['noDist'][0,:int(n_refStim/2)]),'b>', markerfacecolor='b'); 
    ax3.plot(bar_position_wrong[-1]+1,np.nanmean(correct_RT['noDist'][:,int(n_refStim/2):]),'r<', markerfacecolor='r'); 
    ax3.plot(bar_position_wrong[-1]+1,np.nanmean(wrong_RT['noDist'][:,int(n_refStim/2):]),'b<', markerfacecolor='b'); 
    ax3.legend(); 
    ax3.set_xlim([0, bar_position_wrong[-1]+1]); 
    ax3.set_ylim([0, 350]); 
    ax3.set_xticks(bar_position); 
    ax3.set_xticklabels(xlabels); 
    ax3.set_xlabel('Stimulus ID'); 
    ax3.set_ylabel('Reaction time (ms)'); 
    ax3.set_title('No distractor'); 

    plt.tight_layout(); 
    plt.show(); 


    ### NumVary ###
    # col0: stim index
    # col1: category
        # 1 (4-pronged), 2 (3-pronged)
    # col2: number of distractors
    # col3: mode of distractors
        # 1 (all same), 2 (random)    
    # col4: trialCode        
        # 1 (correct), 2 (wrong)        
    # col5: reaction time
    # col6: targ-dist distance (pix)
    # col7: size of distractor (fraction of flankSize)
    # col8: color of distractor
        # 0 (dist color), 1 (targ color), 2 (gray)
        # 3 (dist color CC), 4 (targ color CC), 5 (gray CC)
    # col9: color of target
        # 1 (targ color), 2 (dist color)
    # col10: vary condition number
        # 1 (NumVary), 2 (DistVary), 3 (ColorVary)        
        # 4 (SizeVary), 5 (UncrowdVary), 6 (NumSizeColorVary)       

    selected = np.where(behVec[:,10]==1)[0]; 
    nDist = np.unique(behVec[selected,2]); 

    if len(nDist)>1:

        pCorrect['numVary'] = np.full([len(nDist),n_refStim], np.nan); 
        correct_RT['numVary'] = np.full([len(nDist),n_refStim], np.nan); 
        wrong_RT['numVary'] = np.full([len(nDist),n_refStim], np.nan); 
        pCorrect['numVary_cc'] = np.full([len(nDist),n_refStim], np.nan); 
        correct_RT['numVary_cc'] = np.full([len(nDist),n_refStim], np.nan); 
        wrong_RT['numVary_cc'] = np.full([len(nDist),n_refStim], np.nan); 
        xlabels = []; 
        for i in np.arange(n_refStim):        
            xlabels.append(f'id{i}'); 
            for m in np.arange(len(nDist)):
                all_thisID = np.where((behVec[:,0]==i) &
                                      (behVec[:,2]==nDist[m]) &                
                                      (behVec[:,8]==0) &                                                      
                                      (behVec[:,10]==1))[0]; 
                correct_thisID = np.where((behVec[:,0]==i) &
                                          (behVec[:,2]==nDist[m]) &                
                                          (behVec[:,4]==1) &                                                          
                                          (behVec[:,8]==0) &                                                      
                                          (behVec[:,10]==1))[0]; 
                wrong_thisID = np.where((behVec[:,0]==i) &
                                        (behVec[:,2]==nDist[m]) &                
                                        (behVec[:,4]==2) &                                                          
                                        (behVec[:,8]==0) &                                                      
                                        (behVec[:,10]==1))[0]; 

                # circle control
                all_thisID_cc = np.where((behVec[:,0]==i) &
                                         (behVec[:,2]==nDist[m]) &                
                                         (behVec[:,8]==3) &                                                      
                                         (behVec[:,10]==1))[0]; 
                correct_thisID_cc = np.where((behVec[:,0]==i) &
                                             (behVec[:,2]==nDist[m]) &                
                                             (behVec[:,4]==1) &                                                          
                                             (behVec[:,8]==3) &                                                      
                                             (behVec[:,10]==1))[0]; 
                wrong_thisID_cc = np.where((behVec[:,0]==i) &
                                           (behVec[:,2]==nDist[m]) &                
                                           (behVec[:,4]==2) &                                                          
                                           (behVec[:,8]==3) &                                                      
                                           (behVec[:,10]==1))[0]; 

                pCorrect['numVary'][m,i] = 100 * (len(correct_thisID) / len(all_thisID)); 
                correct_RT['numVary'][m,i] = np.nanmean(behVec[correct_thisID,5]); 
                wrong_RT['numVary'][m,i] = np.nanmean(behVec[wrong_thisID,5]); 

                pCorrect['numVary_cc'][m,i] = 100 * (len(correct_thisID_cc) / len(all_thisID_cc)); 
                correct_RT['numVary_cc'][m,i] = np.nanmean(behVec[correct_thisID_cc,5]); 
                wrong_RT['numVary_cc'][m,i] = np.nanmean(behVec[wrong_thisID_cc,5]); 

        plt.figure(); 
        for m in np.arange(len(nDist)):
            # pCorrect
            plt.subplot(2,len(nDist),m+1); 
            plt.gca().bar(bar_position, pCorrect['numVary'][m,:], facecolor=[0.5,0.5,0.5]);  
            plt.gca().plot([0, bar_position[int(n_refStim/2)-1]+1.5], 
                           [np.nanmean(pCorrect['numVary'][m,:int(n_refStim/2)]), 
                            np.nanmean(pCorrect['numVary'][m,:int(n_refStim/2)])],
                           color='r', linewidth = 2); 
            plt.gca().plot([bar_position[int(n_refStim/2)]-1.5, bar_position[-1]+1.5], 
                           [np.nanmean(pCorrect['numVary'][m,int(n_refStim/2):]), 
                            np.nanmean(pCorrect['numVary'][m,int(n_refStim/2):])],
                           color='r', linewidth = 2); 
            plt.gca().set_xlim([0, bar_position[-1]+1.5]); 
            plt.gca().set_ylim([0, 100]); 
            plt.gca().set_xticks(bar_position); 
            plt.gca().set_xticklabels(xlabels); 
            plt.gca().set_xlabel('Stimulus ID'); 
            plt.gca().set_ylabel('%Correct peformance'); 
            plt.gca().set_title(f'numVary: N = {nDist[m]}'); 

            # reaction time
            plt.subplot(2,len(nDist),m+1+len(nDist)); 
            plt.gca().bar(bar_position_correct, correct_RT['numVary'][m,:],width=0.5, color='r', label='correct');
            plt.gca().bar(bar_position_wrong, wrong_RT['numVary'][m,:], width=0.5, color='b', label='wrong'); 
            plt.gca().plot(0,np.nanmean(correct_RT['numVary'][m,:int(n_refStim/2)]),'r>', markerfacecolor='r'); 
            plt.gca().plot(0,np.nanmean(wrong_RT['numVary'][m,:int(n_refStim/2)]),'b>', markerfacecolor='b'); 
            plt.gca().plot(bar_position_wrong[-1]+1,np.nanmean(correct_RT['numVary'][m,int(n_refStim/2):]),'r<', markerfacecolor='r'); 
            plt.gca().plot(bar_position_wrong[-1]+1,np.nanmean(wrong_RT['numVary'][m,int(n_refStim/2):]),'b<', markerfacecolor='b'); 
            plt.gca().legend(); 
            plt.gca().set_xlim([0, bar_position_wrong[-1]+1]); 
            plt.gca().set_ylim([0, 350]); 
            plt.gca().set_xticks(bar_position); 
            plt.gca().set_xticklabels(xlabels); 
            plt.gca().set_xlabel('Stimulus ID'); 
            plt.gca().set_ylabel('Reaction time (ms)'); 
            plt.gca().set_title(f'numVary: N = {nDist[m]}'); 

        plt.tight_layout(); 
        plt.show(); 

        plt.figure();  # nDist circle control
        for m in np.arange(len(nDist)):
            # pCorrect
            plt.subplot(2,len(nDist),m+1); 
            plt.gca().bar(bar_position, pCorrect['numVary_cc'][m,:], facecolor=[0.5,0.5,0.5]);  
            plt.gca().plot([0, bar_position[int(n_refStim/2)-1]+1.5], 
                           [np.nanmean(pCorrect['numVary_cc'][m,:int(n_refStim/2)]), 
                            np.nanmean(pCorrect['numVary_cc'][m,:int(n_refStim/2)])],
                           color='r', linewidth = 2); 
            plt.gca().plot([bar_position[int(n_refStim/2)]-1.5, bar_position[-1]+1.5], 
                           [np.nanmean(pCorrect['numVary_cc'][m,int(n_refStim/2):]), 
                            np.nanmean(pCorrect['numVary_cc'][m,int(n_refStim/2):])],
                           color='r', linewidth = 2); 
            plt.gca().set_xlim([0, bar_position[-1]+1.5]); 
            plt.gca().set_ylim([0, 100]); 
            plt.gca().set_xticks(bar_position); 
            plt.gca().set_xticklabels(xlabels); 
            plt.gca().set_xlabel('Stimulus ID'); 
            plt.gca().set_ylabel('%Correct peformance'); 
            plt.gca().set_title(f'numVary CC: N = {nDist[m]}'); 

            # reaction time
            plt.subplot(2,len(nDist),m+1+len(nDist)); 
            plt.gca().bar(bar_position_correct, correct_RT['numVary_cc'][m,:],width=0.5, color='r', label='correct');
            plt.gca().bar(bar_position_wrong, wrong_RT['numVary_cc'][m,:], width=0.5, color='b', label='wrong'); 
            plt.gca().plot(0,np.nanmean(correct_RT['numVary_cc'][m,:int(n_refStim/2)]),'r>', markerfacecolor='r'); 
            plt.gca().plot(0,np.nanmean(wrong_RT['numVary_cc'][m,:int(n_refStim/2)]),'b>', markerfacecolor='b'); 
            plt.gca().plot(bar_position_wrong[-1]+1,np.nanmean(correct_RT['numVary_cc'][m,int(n_refStim/2):]),'r<', markerfacecolor='r'); 
            plt.gca().plot(bar_position_wrong[-1]+1,np.nanmean(wrong_RT['numVary_cc'][m,int(n_refStim/2):]),'b<', markerfacecolor='b'); 
            plt.gca().legend(); 
            plt.gca().set_xlim([0, bar_position_wrong[-1]+1]); 
            plt.gca().set_ylim([0, 350]); 
            plt.gca().set_xticks(bar_position); 
            plt.gca().set_xticklabels(xlabels); 
            plt.gca().set_xlabel('Stimulus ID'); 
            plt.gca().set_ylabel('Reaction time (ms)'); 
            plt.gca().set_title(f'numVary CC: N = {nDist[m]}'); 

        plt.tight_layout(); 
        plt.show(); 


    ### DistVary ###
    # col0: stim index
    # col1: category
        # 1 (4-pronged), 2 (3-pronged)
    # col2: number of distractors
    # col3: mode of distractors
        # 1 (all same), 2 (random)    
    # col4: trialCode        
        # 1 (correct), 2 (wrong)        
    # col5: reaction time
    # col6: targ-dist distance (pix)
    # col7: size of distractor (fraction of flankSize)
    # col8: color of distractor
        # 0 (dist color), 1 (targ color), 2 (gray)
        # 3 (dist color CC), 4 (targ color CC), 5 (gray CC)
    # col9: color of target
        # 1 (targ color), 2 (dist color)
    # col10: vary condition number
        # 1 (NumVary), 2 (DistVary), 3 (ColorVary)        
        # 4 (SizeVary), 5 (UncrowdVary), 6 (NumSizeColorVary)       

    selected = np.where(behVec[:,10]==2)[0]; 
    dDist = np.unique(behVec[selected,6]); 

    if len(dDist)>1:

        pCorrect['distVary'] = np.full([len(dDist),n_refStim], np.nan); 
        correct_RT['distVary'] = np.full([len(dDist),n_refStim], np.nan); 
        wrong_RT['distVary'] = np.full([len(dDist),n_refStim], np.nan); 
        pCorrect['distVary_cc'] = np.full([len(dDist),n_refStim], np.nan); 
        correct_RT['distVary_cc'] = np.full([len(dDist),n_refStim], np.nan); 
        wrong_RT['distVary_cc'] = np.full([len(dDist),n_refStim], np.nan); 
        xlabels = []; 
        for i in np.arange(n_refStim):        
            xlabels.append(f'id{i}'); 
            for m in np.arange(len(nDist)):
                all_thisID = np.where((behVec[:,0]==i) &
                                      (behVec[:,6]==dDist[m]) &                
                                      (behVec[:,8]==0) &                                                      
                                      (behVec[:,10]==1))[0]; 
                correct_thisID = np.where((behVec[:,0]==i) &
                                          (behVec[:,6]==dDist[m]) &                
                                          (behVec[:,4]==1) &                                                          
                                          (behVec[:,8]==0) &                                                      
                                          (behVec[:,10]==1))[0]; 
                wrong_thisID = np.where((behVec[:,0]==i) &
                                        (behVec[:,6]==dDist[m]) &                
                                        (behVec[:,4]==2) &                                                          
                                        (behVec[:,8]==0) &                                                      
                                        (behVec[:,10]==1))[0]; 

                # circle control
                all_thisID_cc = np.where((behVec[:,0]==i) &
                                         (behVec[:,6]==dDist[m]) &                
                                         (behVec[:,8]==3) &                                                      
                                         (behVec[:,10]==1))[0]; 
                correct_thisID_cc = np.where((behVec[:,0]==i) &
                                             (behVec[:,6]==dDist[m]) &                
                                             (behVec[:,4]==1) &                                                          
                                             (behVec[:,8]==3) &                                                      
                                             (behVec[:,10]==1))[0]; 
                wrong_thisID_cc = np.where((behVec[:,0]==i) &
                                           (behVec[:,6]==dDist[m]) &                
                                           (behVec[:,4]==2) &                                                          
                                           (behVec[:,8]==3) &                                                      
                                           (behVec[:,10]==1))[0]; 

                pCorrect['distVary'][m,i] = 100 * (len(correct_thisID) / len(all_thisID)); 
                correct_RT['distVary'][m,i] = np.nanmean(behVec[correct_thisID,5]); 
                wrong_RT['distVary'][m,i] = np.nanmean(behVec[wrong_thisID,5]); 

                pCorrect['distVary_cc'][m,i] = 100 * (len(correct_thisID_cc) / len(all_thisID_cc)); 
                correct_RT['distVary_cc'][m,i] = np.nanmean(behVec[correct_thisID_cc,5]); 
                wrong_RT['distVary_cc'][m,i] = np.nanmean(behVec[wrong_thisID_cc,5]); 

        plt.figure(); 
        for m in np.arange(len(dDist)):
            # pCorrect
            plt.subplot(2,len(dDist),m+1); 
            plt.gca().bar(bar_position, pCorrect['distVary'][m,:], facecolor=[0.5,0.5,0.5]);  
            plt.gca().plot([0, bar_position[int(n_refStim/2)-1]+1.5], 
                           [np.nanmean(pCorrect['distVary'][m,:int(n_refStim/2)]), 
                            np.nanmean(pCorrect['distVary'][m,:int(n_refStim/2)])],
                           color='r', linewidth = 2); 
            plt.gca().plot([bar_position[int(n_refStim/2)]-1.5, bar_position[-1]+1.5], 
                           [np.nanmean(pCorrect['distVary'][m,int(n_refStim/2):]), 
                            np.nanmean(pCorrect['distVary'][m,int(n_refStim/2):])],
                           color='r', linewidth = 2); 
            plt.gca().set_xlim([0, bar_position[-1]+1.5]); 
            plt.gca().set_ylim([0, 100]); 
            plt.gca().set_xticks(bar_position); 
            plt.gca().set_xticklabels(xlabels); 
            plt.gca().set_xlabel('Stimulus ID'); 
            plt.gca().set_ylabel('%Correct peformance'); 
            plt.gca().set_title(f'distVary: D = {dDist[m]}'); 

            # reaction time
            plt.subplot(2,len(dDist),m+1+len(dDist)); 
            plt.gca().bar(bar_position_correct, correct_RT['distVary'][m,:],width=0.5, color='r', label='correct');
            plt.gca().bar(bar_position_wrong, wrong_RT['distVary'][m,:], width=0.5, color='b', label='wrong'); 
            plt.gca().plot(0,np.nanmean(correct_RT['distVary'][m,:int(n_refStim/2)]),'r>', markerfacecolor='r'); 
            plt.gca().plot(0,np.nanmean(wrong_RT['distVary'][m,:int(n_refStim/2)]),'b>', markerfacecolor='b'); 
            plt.gca().plot(bar_position_wrong[-1]+1,np.nanmean(correct_RT['distVary'][m,int(n_refStim/2):]),'r<', markerfacecolor='r'); 
            plt.gca().plot(bar_position_wrong[-1]+1,np.nanmean(wrong_RT['distVary'][m,int(n_refStim/2):]),'b<', markerfacecolor='b'); 
            plt.gca().legend(); 
            plt.gca().set_xlim([0, bar_position_wrong[-1]+1]); 
            plt.gca().set_ylim([0, 350]); 
            plt.gca().set_xticks(bar_position); 
            plt.gca().set_xticklabels(xlabels); 
            plt.gca().set_xlabel('Stimulus ID'); 
            plt.gca().set_ylabel('Reaction time (ms)'); 
            plt.gca().set_title(f'distVary: D = {dDist[m]}'); 

        plt.tight_layout(); 
        plt.show(); 

        plt.figure();   # circle control
        for m in np.arange(len(dDist)):
            # pCorrect
            plt.subplot(2,len(dDist),m+1); 
            plt.gca().bar(bar_position, pCorrect['distVary_cc'][m,:], facecolor=[0.5,0.5,0.5]);  
            plt.gca().plot([0, bar_position[int(n_refStim/2)-1]+1.5], 
                           [np.nanmean(pCorrect['distVary_cc'][m,:int(n_refStim/2)]), 
                            np.nanmean(pCorrect['distVary_cc'][m,:int(n_refStim/2)])],
                           color='r', linewidth = 2); 
            plt.gca().plot([bar_position[int(n_refStim/2)]-1.5, bar_position[-1]+1.5], 
                           [np.nanmean(pCorrect['distVary_cc'][m,int(n_refStim/2):]), 
                            np.nanmean(pCorrect['distVary_cc'][m,int(n_refStim/2):])],
                           color='r', linewidth = 2); 
            plt.gca().set_xlim([0, bar_position[-1]+1.5]); 
            plt.gca().set_ylim([0, 100]); 
            plt.gca().set_xticks(bar_position); 
            plt.gca().set_xticklabels(xlabels); 
            plt.gca().set_xlabel('Stimulus ID'); 
            plt.gca().set_ylabel('%Correct peformance'); 
            plt.gca().set_title(f'distVary CC: D = {dDist[m]}'); 

            # reaction time
            plt.subplot(2,len(dDist),m+1+len(dDist)); 
            plt.gca().bar(bar_position_correct, correct_RT['distVary_cc'][m,:],width=0.5, color='r', label='correct');
            plt.gca().bar(bar_position_wrong, wrong_RT['distVary_cc'][m,:], width=0.5, color='b', label='wrong'); 
            plt.gca().plot(0,np.nanmean(correct_RT['distVary_cc'][m,:int(n_refStim/2)]),'r>', markerfacecolor='r'); 
            plt.gca().plot(0,np.nanmean(wrong_RT['distVary_cc'][m,:int(n_refStim/2)]),'b>', markerfacecolor='b'); 
            plt.gca().plot(bar_position_wrong[-1]+1,np.nanmean(correct_RT['distVary_cc'][m,int(n_refStim/2):]),'r<', markerfacecolor='r'); 
            plt.gca().plot(bar_position_wrong[-1]+1,np.nanmean(wrong_RT['distVary_cc'][m,int(n_refStim/2):]),'b<', markerfacecolor='b'); 
            plt.gca().legend(); 
            plt.gca().set_xlim([0, bar_position_wrong[-1]+1]); 
            plt.gca().set_ylim([0, 350]); 
            plt.gca().set_xticks(bar_position); 
            plt.gca().set_xticklabels(xlabels); 
            plt.gca().set_xlabel('Stimulus ID'); 
            plt.gca().set_ylabel('Reaction time (ms)'); 
            plt.gca().set_title(f'distVary CC: D = {dDist[m]}'); 

        plt.tight_layout(); 
        plt.show();         


    ### ColorVary ###
    # col0: stim index
    # col1: category
        # 1 (4-pronged), 2 (3-pronged)
    # col2: number of distractors
    # col3: mode of distractors
        # 1 (all same), 2 (random)    
    # col4: trialCode        
        # 1 (correct), 2 (wrong)        
    # col5: reaction time
    # col6: targ-dist distance (pix)
    # col7: size of distractor (fraction of flankSize)
    # col8: color of distractor
        # 0 (dist color), 1 (targ color), 2 (gray)
        # 3 (dist color CC), 4 (targ color CC), 5 (gray CC)
    # col9: color of target
        # 1 (targ color), 2 (dist color)
    # col10: vary condition number
        # 1 (NumVary), 2 (DistVary), 3 (ColorVary)        
        # 4 (SizeVary), 5 (UncrowdVary), 6 (NumSizeColorVary)       

    color_pair = np.array([[0,1],[1,1],[2,1],[1,2]]); 
    cDist = []; 
    for i in range(4):
        colorNow = np.where((behVec[:,8]==color_pair[i,0]) &
                            (behVec[:,9]==color_pair[i,1]) &
                            (behVec[:,2]>0) &         
                            (behVec[:,10]==3))[0]; 
        if len(colorNow)>0:
            cDist.append(i); 

    if len(cDist)>1:

        pCorrect['colorVary'] = np.full([len(cDist),n_refStim], np.nan); 
        correct_RT['colorVary'] = np.full([len(cDist),n_refStim], np.nan); 
        wrong_RT['colorVary'] = np.full([len(cDist),n_refStim], np.nan); 
        pCorrect['colorVary_cc'] = np.full([len(cDist),n_refStim], np.nan); 
        correct_RT['colorVary_cc'] = np.full([len(cDist),n_refStim], np.nan); 
        wrong_RT['colorVary_cc'] = np.full([len(cDist),n_refStim], np.nan); 
        xlabels = []; 
        for i in np.arange(n_refStim):        
            xlabels.append(f'id{i}'); 
            for m in np.arange(len(nDist)):
                all_thisID = np.where((behVec[:,0]==i) &
                                      (behVec[:,2]>0) &
                                      (behVec[:,6]==dDist[m]) &                
                                      (behVec[:,8]==color_pair[cDist[m],0]) &                                                      
                                      (behVec[:,9]==color_pair[cDist[m],1]) &                                                                                            
                                      (behVec[:,10]==3))[0]; 
                correct_thisID = np.where((behVec[:,0]==i) &
                                          (behVec[:,2]>0) &
                                          (behVec[:,4]==1) &                                          
                                          (behVec[:,6]==dDist[m]) &                
                                          (behVec[:,8]==color_pair[cDist[m],0]) &                                                      
                                          (behVec[:,9]==color_pair[cDist[m],1]) &                                                                                            
                                          (behVec[:,10]==3))[0]; 
                wrong_thisID = np.where((behVec[:,0]==i) &
                                        (behVec[:,2]>0) &
                                        (behVec[:,4]==2) &                                          
                                        (behVec[:,6]==dDist[m]) &                
                                        (behVec[:,8]==color_pair[cDist[m],0]) &                                                      
                                        (behVec[:,9]==color_pair[cDist[m],1]) &                                                                                            
                                        (behVec[:,10]==3))[0]; 

                # circle control
                all_thisID_cc = np.where((behVec[:,0]==i) &
                                         (behVec[:,2]>0) &
                                         (behVec[:,6]==dDist[m]) &                
                                         (behVec[:,8]==color_pair[cDist[m],0]+3) &                                                      
                                         (behVec[:,9]==color_pair[cDist[m],1]) &                                                                                            
                                         (behVec[:,10]==3))[0]; 
                correct_thisID_cc = np.where((behVec[:,0]==i) &
                                             (behVec[:,2]>0) &
                                             (behVec[:,4]==1) &                                          
                                             (behVec[:,6]==dDist[m]) &                
                                             (behVec[:,8]==color_pair[cDist[m],0]+3) &                                                      
                                             (behVec[:,9]==color_pair[cDist[m],1]) &                                                                                            
                                             (behVec[:,10]==3))[0]; 
                wrong_thisID_cc = np.where((behVec[:,0]==i) &
                                           (behVec[:,2]>0) &
                                           (behVec[:,4]==2) &                                          
                                           (behVec[:,6]==dDist[m]) &                
                                           (behVec[:,8]==color_pair[cDist[m],0]+3) &                                                      
                                           (behVec[:,9]==color_pair[cDist[m],1]) &                                                                                            
                                           (behVec[:,10]==3))[0]; 

                pCorrect['colorVary'][m,i] = 100 * (len(correct_thisID) / len(all_thisID)); 
                correct_RT['colorVary'][m,i] = np.nanmean(behVec[correct_thisID,5]); 
                wrong_RT['colorVary'][m,i] = np.nanmean(behVec[wrong_thisID,5]); 

                pCorrect['colorVary_cc'][m,i] = 100 * (len(correct_thisID_cc) / len(all_thisID_cc)); 
                correct_RT['colorVary_cc'][m,i] = np.nanmean(behVec[correct_thisID_cc,5]); 
                wrong_RT['colorVary_cc'][m,i] = np.nanmean(behVec[wrong_thisID_cc,5]);             

        plt.figure(); 
        for m in np.arange(len(cDist)):
            # pCorrect
            plt.subplot(2,len(cDist),m+1); 
            plt.gca().bar(bar_position, pCorrect['colorVary'][m,:], facecolor=[0.5,0.5,0.5]);  
            plt.gca().plot([0, bar_position[int(n_refStim/2)-1]+1.5], 
                           [np.nanmean(pCorrect['colorVary'][m,:int(n_refStim/2)]), 
                            np.nanmean(pCorrect['colorVary'][m,:int(n_refStim/2)])],
                           color='r', linewidth = 2); 
            plt.gca().plot([bar_position[int(n_refStim/2)]-1.5, bar_position[-1]+1.5], 
                           [np.nanmean(pCorrect['colorVary'][m,int(n_refStim/2):]), 
                            np.nanmean(pCorrect['colorVary'][m,int(n_refStim/2):])],
                           color='r', linewidth = 2); 
            plt.gca().set_xlim([0, bar_position[-1]+1.5]); 
            plt.gca().set_ylim([0, 100]); 
            plt.gca().set_xticks(bar_position); 
            plt.gca().set_xticklabels(xlabels); 
            plt.gca().set_xlabel('Stimulus ID'); 
            plt.gca().set_ylabel('%Correct peformance'); 
            plt.gca().set_title(f'colorVary: color-pair = {color_pair[cDist[m],0]}-{color_pair[cDist[m],1]}'); 

            # reaction time
            plt.subplot(2,len(cDist),m+1+len(cDist)); 
            plt.gca().bar(bar_position_correct, correct_RT['colorVary'][m,:],width=0.5, color='r', label='correct');
            plt.gca().bar(bar_position_wrong, wrong_RT['colorVary'][m,:], width=0.5, color='b', label='wrong'); 
            plt.gca().plot(0,np.nanmean(correct_RT['colorVary'][m,:int(n_refStim/2)]),'r>', markerfacecolor='r'); 
            plt.gca().plot(0,np.nanmean(wrong_RT['colorVary'][m,:int(n_refStim/2)]),'b>', markerfacecolor='b'); 
            plt.gca().plot(bar_position_wrong[-1]+1,np.nanmean(correct_RT['colorVary'][m,int(n_refStim/2):]),'r<', markerfacecolor='r'); 
            plt.gca().plot(bar_position_wrong[-1]+1,np.nanmean(wrong_RT['colorVary'][m,int(n_refStim/2):]),'b<', markerfacecolor='b'); 
            plt.gca().legend(); 
            plt.gca().set_xlim([0, bar_position_wrong[-1]+1]); 
            plt.gca().set_ylim([0, 350]); 
            plt.gca().set_xticks(bar_position); 
            plt.gca().set_xticklabels(xlabels); 
            plt.gca().set_xlabel('Stimulus ID'); 
            plt.gca().set_ylabel('Reaction time (ms)'); 
            plt.gca().set_title(f'colorVary: color-pair = {color_pair[cDist[m],0]}-{color_pair[cDist[m],1]}'); 

        plt.tight_layout(); 
        plt.show(); 

        plt.figure();   # circle control
        for m in np.arange(len(cDist)):
            # pCorrect
            plt.subplot(2,len(cDist),m+1); 
            plt.gca().bar(bar_position, pCorrect['colorVary_cc'][m,:], facecolor=[0.5,0.5,0.5]);  
            plt.gca().plot([0, bar_position[int(n_refStim/2)-1]+1.5], 
                           [np.nanmean(pCorrect['colorVary_cc'][m,:int(n_refStim/2)]), 
                            np.nanmean(pCorrect['colorVary_cc'][m,:int(n_refStim/2)])],
                           color='r', linewidth = 2); 
            plt.gca().plot([bar_position[int(n_refStim/2)]-1.5, bar_position[-1]+1.5], 
                           [np.nanmean(pCorrect['colorVary_cc'][m,int(n_refStim/2):]), 
                            np.nanmean(pCorrect['colorVary_cc'][m,int(n_refStim/2):])],
                           color='r', linewidth = 2); 
            plt.gca().set_xlim([0, bar_position[-1]+1.5]); 
            plt.gca().set_ylim([0, 100]); 
            plt.gca().set_xticks(bar_position); 
            plt.gca().set_xticklabels(xlabels); 
            plt.gca().set_xlabel('Stimulus ID'); 
            plt.gca().set_ylabel('%Correct peformance'); 
            plt.gca().set_title(f'colorVary CC: color-pair = {color_pair[cDist[m],0]}-{color_pair[cDist[m],1]}'); 

            # reaction time
            plt.subplot(2,len(cDist),m+1+len(cDist)); 
            plt.gca().bar(bar_position_correct, correct_RT['colorVary_cc'][m,:],width=0.5, color='r', label='correct');
            plt.gca().bar(bar_position_wrong, wrong_RT['colorVary_cc'][m,:], width=0.5, color='b', label='wrong'); 
            plt.gca().plot(0,np.nanmean(correct_RT['colorVary_cc'][m,:int(n_refStim/2)]),'r>', markerfacecolor='r'); 
            plt.gca().plot(0,np.nanmean(wrong_RT['colorVary_cc'][m,:int(n_refStim/2)]),'b>', markerfacecolor='b'); 
            plt.gca().plot(bar_position_wrong[-1]+1,np.nanmean(correct_RT['colorVary_cc'][m,int(n_refStim/2):]),'r<', markerfacecolor='r'); 
            plt.gca().plot(bar_position_wrong[-1]+1,np.nanmean(wrong_RT['colorVary_cc'][m,int(n_refStim/2):]),'b<', markerfacecolor='b'); 
            plt.gca().legend(); 
            plt.gca().set_xlim([0, bar_position_wrong[-1]+1]); 
            plt.gca().set_ylim([0, 350]); 
            plt.gca().set_xticks(bar_position); 
            plt.gca().set_xticklabels(xlabels); 
            plt.gca().set_xlabel('Stimulus ID'); 
            plt.gca().set_ylabel('Reaction time (ms)'); 
            plt.gca().set_title(f'colorVary CC: color-pair = {color_pair[cDist[m],0]}-{color_pair[cDist[m],1]}'); 

        plt.tight_layout(); 
        plt.show();         


    ### SizeVary ###
    # col0: stim index
    # col1: category
        # 1 (4-pronged), 2 (3-pronged)
    # col2: number of distractors
    # col3: mode of distractors
        # 1 (all same), 2 (random)    
    # col4: trialCode        
        # 1 (correct), 2 (wrong)        
    # col5: reaction time
    # col6: targ-dist distance (pix)
    # col7: size of distractor (fraction of flankSize)
    # col8: color of distractor
        # 0 (dist color), 1 (targ color), 2 (gray)
        # 3 (dist color CC), 4 (targ color CC), 5 (gray CC)
    # col9: color of target
        # 1 (targ color), 2 (dist color)
    # col10: vary condition number
        # 1 (NumVary), 2 (DistVary), 3 (ColorVary)        
        # 4 (SizeVary), 5 (UncrowdVary), 6 (NumSizeColorVary)       

    selected = np.where(behVec[:,10]==4)[0]; 
    sDist = np.unique(behVec[selected,7]); 

    if len(sDist)>1:

        pCorrect['sizeVary'] = np.full([len(sDist),n_refStim], np.nan); 
        correct_RT['sizeVary'] = np.full([len(sDist),n_refStim], np.nan); 
        wrong_RT['sizeVary'] = np.full([len(sDist),n_refStim], np.nan); 
        pCorrect['sizeVary_cc'] = np.full([len(sDist),n_refStim], np.nan); 
        correct_RT['sizeVary_cc'] = np.full([len(sDist),n_refStim], np.nan); 
        wrong_RT['sizeVary_cc'] = np.full([len(sDist),n_refStim], np.nan); 
        xlabels = []; 
        for i in np.arange(n_refStim):        
            xlabels.append(f'id{i}'); 
            for m in np.arange(len(sDist)):
                all_thisID = np.where((behVec[:,0]==i) &
                                      (behVec[:,2]>0) &                                
                                      (behVec[:,7]==sDist[m]) &                
                                      (behVec[:,8]==0) &                                                      
                                      (behVec[:,10]==4))[0]; 
                correct_thisID = np.where((behVec[:,0]==i) &
                                          (behVec[:,2]>0) &          
                                          (behVec[:,4]==1) &                                
                                          (behVec[:,7]==sDist[m]) &                
                                          (behVec[:,8]==0) &                                                      
                                          (behVec[:,10]==4))[0]; 
                wrong_thisID = np.where((behVec[:,0]==i) &
                                        (behVec[:,2]>0) &          
                                        (behVec[:,4]==2) &                                
                                        (behVec[:,7]==sDist[m]) &                
                                        (behVec[:,8]==0) &                                                      
                                        (behVec[:,10]==4))[0]; 

                # circle control
                all_thisID_cc = np.where((behVec[:,0]==i) &
                                         (behVec[:,2]>0) &                                
                                         (behVec[:,7]==sDist[m]) &                
                                         (behVec[:,8]==3) &                                                      
                                         (behVec[:,10]==4))[0]; 
                correct_thisID_cc = np.where((behVec[:,0]==i) &
                                             (behVec[:,2]>0) &          
                                             (behVec[:,4]==1) &                                
                                             (behVec[:,7]==sDist[m]) &                
                                             (behVec[:,8]==3) &                                                      
                                             (behVec[:,10]==4))[0]; 
                wrong_thisID_cc = np.where((behVec[:,0]==i) &
                                           (behVec[:,2]>0) &          
                                           (behVec[:,4]==2) &                                
                                           (behVec[:,7]==sDist[m]) &                
                                           (behVec[:,8]==3) &                                                      
                                           (behVec[:,10]==4))[0]; 

                pCorrect['sizeVary'][m,i] = 100 * (len(correct_thisID) / len(all_thisID)); 
                correct_RT['sizeVary'][m,i] = np.nanmean(behVec[correct_thisID,5]); 
                wrong_RT['sizeVary'][m,i] = np.nanmean(behVec[wrong_thisID,5]); 

                pCorrect['sizeVary_cc'][m,i] = 100 * (len(correct_thisID_cc) / len(all_thisID_cc)); 
                correct_RT['sizeVary_cc'][m,i] = np.nanmean(behVec[correct_thisID_cc,5]); 
                wrong_RT['sizeVary_cc'][m,i] = np.nanmean(behVec[wrong_thisID_cc,5]); 

        plt.figure(); 
        for m in np.arange(len(sDist)):
            # pCorrect
            plt.subplot(2,len(sDist),m+1); 
            plt.gca().bar(bar_position, pCorrect['sizeVary'][m,:], facecolor=[0.5,0.5,0.5]);  
            plt.gca().plot([0, bar_position[int(n_refStim/2)-1]+1.5], 
                           [np.nanmean(pCorrect['sizeVary'][m,:int(n_refStim/2)]), 
                            np.nanmean(pCorrect['sizeVary'][m,:int(n_refStim/2)])],
                           color='r', linewidth = 2); 
            plt.gca().plot([bar_position[int(n_refStim/2)]-1.5, bar_position[-1]+1.5], 
                           [np.nanmean(pCorrect['sizeVary'][m,int(n_refStim/2):]), 
                            np.nanmean(pCorrect['sizeVary'][m,int(n_refStim/2):])],
                           color='r', linewidth = 2); 
            plt.gca().set_xlim([0, bar_position[-1]+1.5]); 
            plt.gca().set_ylim([0, 100]); 
            plt.gca().set_xticks(bar_position); 
            plt.gca().set_xticklabels(xlabels); 
            plt.gca().set_xlabel('Stimulus ID'); 
            plt.gca().set_ylabel('%Correct peformance'); 
            plt.gca().set_title(f'sizeVary: {sDist[m]}'); 

            # reaction time
            plt.subplot(2,len(sDist),m+1+len(sDist)); 
            plt.gca().bar(bar_position_correct, correct_RT['sizeVary'][m,:],width=0.5, color='r', label='correct');
            plt.gca().bar(bar_position_wrong, wrong_RT['sizeVary'][m,:], width=0.5, color='b', label='wrong'); 
            plt.gca().plot(0,np.nanmean(correct_RT['sizeVary'][m,:int(n_refStim/2)]),'r>', markerfacecolor='r'); 
            plt.gca().plot(0,np.nanmean(wrong_RT['sizeVary'][m,:int(n_refStim/2)]),'b>', markerfacecolor='b'); 
            plt.gca().plot(bar_position_wrong[-1]+1,np.nanmean(correct_RT['sizeVary'][m,int(n_refStim/2):]),'r<', markerfacecolor='r'); 
            plt.gca().plot(bar_position_wrong[-1]+1,np.nanmean(wrong_RT['sizeVary'][m,int(n_refStim/2):]),'b<', markerfacecolor='b'); 
            plt.gca().legend(); 
            plt.gca().set_xlim([0, bar_position_wrong[-1]+1]); 
            plt.gca().set_ylim([0, 350]); 
            plt.gca().set_xticks(bar_position); 
            plt.gca().set_xticklabels(xlabels); 
            plt.gca().set_xlabel('Stimulus ID'); 
            plt.gca().set_ylabel('Reaction time (ms)'); 
            plt.gca().set_title(f'sizeVary: {sDist[m]}'); 

        plt.tight_layout(); 
        plt.show(); 

        plt.figure();   # circle control
        for m in np.arange(len(sDist)):
            # pCorrect
            plt.subplot(2,len(sDist),m+1); 
            plt.gca().bar(bar_position, pCorrect['sizeVary_cc'][m,:], facecolor=[0.5,0.5,0.5]);  
            plt.gca().plot([0, bar_position[int(n_refStim/2)-1]+1.5], 
                           [np.nanmean(pCorrect['sizeVary_cc'][m,:int(n_refStim/2)]), 
                            np.nanmean(pCorrect['sizeVary_cc'][m,:int(n_refStim/2)])],
                           color='r', linewidth = 2); 
            plt.gca().plot([bar_position[int(n_refStim/2)]-1.5, bar_position[-1]+1.5], 
                           [np.nanmean(pCorrect['sizeVary_cc'][m,int(n_refStim/2):]), 
                            np.nanmean(pCorrect['sizeVary_cc'][m,int(n_refStim/2):])],
                           color='r', linewidth = 2); 
            plt.gca().set_xlim([0, bar_position[-1]+1.5]); 
            plt.gca().set_ylim([0, 100]); 
            plt.gca().set_xticks(bar_position); 
            plt.gca().set_xticklabels(xlabels); 
            plt.gca().set_xlabel('Stimulus ID'); 
            plt.gca().set_ylabel('%Correct peformance'); 
            plt.gca().set_title(f'sizeVary CC: {sDist[m]}'); 

            # reaction time
            plt.subplot(2,len(sDist),m+1+len(sDist)); 
            plt.gca().bar(bar_position_correct, correct_RT['sizeVary_cc'][m,:],width=0.5, color='r', label='correct');
            plt.gca().bar(bar_position_wrong, wrong_RT['sizeVary_cc'][m,:], width=0.5, color='b', label='wrong'); 
            plt.gca().plot(0,np.nanmean(correct_RT['sizeVary_cc'][m,:int(n_refStim/2)]),'r>', markerfacecolor='r'); 
            plt.gca().plot(0,np.nanmean(wrong_RT['sizeVary_cc'][m,:int(n_refStim/2)]),'b>', markerfacecolor='b'); 
            plt.gca().plot(bar_position_wrong[-1]+1,np.nanmean(correct_RT['sizeVary_cc'][m,int(n_refStim/2):]),'r<', markerfacecolor='r'); 
            plt.gca().plot(bar_position_wrong[-1]+1,np.nanmean(wrong_RT['sizeVary_cc'][m,int(n_refStim/2):]),'b<', markerfacecolor='b'); 
            plt.gca().legend(); 
            plt.gca().set_xlim([0, bar_position_wrong[-1]+1]); 
            plt.gca().set_ylim([0, 350]); 
            plt.gca().set_xticks(bar_position); 
            plt.gca().set_xticklabels(xlabels); 
            plt.gca().set_xlabel('Stimulus ID'); 
            plt.gca().set_ylabel('Reaction time (ms)'); 
            plt.gca().set_title(f'sizeVary CC: {sDist[m]}'); 

        plt.tight_layout(); 
        plt.show(); 


    ### UncrowdVary ###
    # col0: stim index
    # col1: category
        # 1 (4-pronged), 2 (3-pronged)
    # col2: number of distractors
    # col3: mode of distractors
        # 1 (all same), 2 (random)    
    # col4: trialCode        
        # 1 (correct), 2 (wrong)        
    # col5: reaction time
    # col6: targ-dist distance (pix)
    # col7: size of distractor (fraction of flankSize)
    # col8: color of distractor
        # 0 (dist color), 1 (targ color), 2 (gray)
        # 3 (dist color CC), 4 (targ color CC), 5 (gray CC)
    # col9: color of target
        # 1 (targ color), 2 (dist color)
    # col10: vary condition number
        # 1 (NumVary), 2 (DistVary), 3 (ColorVary)        
        # 4 (SizeVary), 5 (UncrowdVary), 6 (NumSizeColorVary)      

    selected = np.where((behVec[:,2]>0) &
                        (behVec[:,10]==5))[0]; 
    uDist = np.unique(behVec[selected,2]); 

    if len(uDist)>=1:

        pCorrect['uncrowdVary'] = np.full([len(sDist),n_refStim], np.nan); 
        correct_RT['uncrowdVary'] = np.full([len(sDist),n_refStim], np.nan); 
        wrong_RT['uncrowdVary'] = np.full([len(sDist),n_refStim], np.nan); 

        xlabels = []; 
        for i in np.arange(n_refStim):        
            xlabels.append(f'id{i}'); 
            for m in np.arange(len(uDist)):
                all_thisID = np.where((behVec[:,0]==i) &
                                      (behVec[:,2]==uDist[m]) &                
                                      (behVec[:,10]==5))[0]; 
                correct_thisID = np.where((behVec[:,0]==i) &
                                          (behVec[:,2]==uDist[m]) &    
                                          (behVec[:,4]==1) &                                                          
                                          (behVec[:,10]==5))[0]; 
                wrong_thisID = np.where((behVec[:,0]==i) &
                                        (behVec[:,2]==uDist[m]) &    
                                        (behVec[:,4]==2) &                                                          
                                        (behVec[:,10]==5))[0]; 

                pCorrect['uncrowdVary'][m,i] = 100 * (len(correct_thisID) / len(all_thisID)); 
                correct_RT['uncrowdVary'][m,i] = np.nanmean(behVec[correct_thisID,5]); 
                wrong_RT['uncrowdVary'][m,i] = np.nanmean(behVec[wrong_thisID,5]); 

        plt.figure(); 
        for m in np.arange(len(uDist)):
            # pCorrect
            if len(uDist)==1:
                plt.subplot(2,2,1);
            else:
                plt.subplot(2,len(uDist),m+1); 
            plt.gca().bar(bar_position, pCorrect['uncrowdVary'][m,:], facecolor=[0.5,0.5,0.5]);  
            plt.gca().plot([0, bar_position[int(n_refStim/2)-1]+1.5], 
                           [np.nanmean(pCorrect['uncrowdVary'][m,:int(n_refStim/2)]), 
                            np.nanmean(pCorrect['uncrowdVary'][m,:int(n_refStim/2)])],
                           color='r', linewidth = 2); 
            plt.gca().plot([bar_position[int(n_refStim/2)]-1.5, bar_position[-1]+1.5], 
                           [np.nanmean(pCorrect['uncrowdVary'][m,int(n_refStim/2):]), 
                            np.nanmean(pCorrect['uncrowdVary'][m,int(n_refStim/2):])],
                           color='r', linewidth = 2); 
            plt.gca().set_xlim([0, bar_position[-1]+1.5]); 
            plt.gca().set_ylim([0, 100]); 
            plt.gca().set_xticks(bar_position); 
            plt.gca().set_xticklabels(xlabels); 
            plt.gca().set_xlabel('Stimulus ID'); 
            plt.gca().set_ylabel('%Correct peformance'); 
            plt.gca().set_title(f'uncrowdVary: {uDist[m]}'); 

            # reaction time
            if len(uDist)==1:
                plt.subplot(2,2,3);
            else:
                plt.subplot(2,len(uDist),m+1+len(uDist)); 
            plt.gca().bar(bar_position_correct, correct_RT['uncrowdVary'][m,:],width=0.5, color='r', label='correct');
            plt.gca().bar(bar_position_wrong, wrong_RT['uncrowdVary'][m,:], width=0.5, color='b', label='wrong'); 
            plt.gca().plot(0,np.nanmean(correct_RT['uncrowdVary'][m,:int(n_refStim/2)]),'r>', markerfacecolor='r'); 
            plt.gca().plot(0,np.nanmean(wrong_RT['uncrowdVary'][m,:int(n_refStim/2)]),'b>', markerfacecolor='b'); 
            plt.gca().plot(bar_position_wrong[-1]+1,np.nanmean(correct_RT['uncrowdVary'][m,int(n_refStim/2):]),'r<', markerfacecolor='r'); 
            plt.gca().plot(bar_position_wrong[-1]+1,np.nanmean(wrong_RT['uncrowdVary'][m,int(n_refStim/2):]),'b<', markerfacecolor='b'); 
            plt.gca().legend(); 
            plt.gca().set_xlim([0, bar_position_wrong[-1]+1]); 
            plt.gca().set_ylim([0, 350]); 
            plt.gca().set_xticks(bar_position); 
            plt.gca().set_xticklabels(xlabels); 
            plt.gca().set_xlabel('Stimulus ID'); 
            plt.gca().set_ylabel('Reaction time (ms)'); 
            plt.gca().set_title(f'uncrowdVary: {uDist[m]}'); 

        plt.tight_layout(); 
        plt.show(); 


    ### NumSizeColorVary ###
    # col0: stim index
    # col1: category
        # 1 (4-pronged), 2 (3-pronged)
    # col2: number of distractors
    # col3: mode of distractors
        # 1 (all same), 2 (random)    
    # col4: trialCode        
        # 1 (correct), 2 (wrong)        
    # col5: reaction time
    # col6: targ-dist distance (pix)
    # col7: size of distractor (fraction of flankSize)
    # col8: color of distractor
        # 0 (dist color), 1 (targ color), 2 (gray)
        # 3 (dist color CC), 4 (targ color CC), 5 (gray CC)
    # col9: color of target
        # 1 (targ color), 2 (dist color)
    # col10: vary condition number
        # 1 (NumVary), 2 (DistVary), 3 (ColorVary)        
        # 4 (SizeVary), 5 (UncrowdVary), 6 (NumSizeColorVary)      

    selected = np.where(behVec[:,10]==6)[0]; 
    # Number effect
    nDist = np.unique(behVec[selected,2]); 
    # Size effect
    sDist = np.unique(behVec[selected,7]); 
    # Color effect
    color_pair = np.array([[0,1],[1,1],[2,1],[1,2]]); 
    cDist = []; 
    for i in range(4):
        colorNow = np.where((behVec[selected,8]==color_pair[i,0]) &
                            (behVec[selected,9]==color_pair[i,1]))[0]; 
        if len(colorNow)>0:
            cDist.append(i); 

    if len(selected)>0:
        pCorrect['multiVary'] = np.full([len(nDist),n_refStim,len(sDist),len(cDist)], np.nan); 
        correct_RT['multiVary'] = np.full([len(nDist),n_refStim,len(sDist),len(cDist)], np.nan); 
        wrong_RT['multiVary'] = np.full([len(nDist),n_refStim,len(sDist),len(cDist)], np.nan); 

        pCorrect['multiVary_cc'] = np.full([len(nDist),n_refStim,len(sDist),len(cDist)], np.nan); 
        correct_RT['multiVary_cc'] = np.full([len(nDist),n_refStim,len(sDist),len(cDist)], np.nan); 
        wrong_RT['multiVary_cc'] = np.full([len(nDist),n_refStim,len(sDist),len(cDist)], np.nan); 

        for s in np.arange(len(sDist)):   # size
            for c in np.arange(len(cDist)):   # color
                xlabels = [];             
                for i in np.arange(n_refStim):   # reference shape                 
                    xlabels.append(f'id{i}'); 
                    for m in np.arange(len(nDist)):   # number
                        all_thisID = np.where((behVec[:,0]==i) &
                                            (behVec[:,2]==nDist[m]) &                
                                            (behVec[:,7]==sDist[s]) &                                                          
                                            (behVec[:,8]==color_pair[cDist[c],0]) &                                                          
                                            (behVec[:,9]==color_pair[cDist[c],1]) &                                                                                                                                              
                                            (behVec[:,10]==6))[0]; 

                        correct_thisID = np.where((behVec[:,0]==i) &
                                                (behVec[:,2]==nDist[m]) &                
                                                (behVec[:,4]==1) &                                                              
                                                (behVec[:,7]==sDist[s]) &                                                          
                                                (behVec[:,8]==color_pair[cDist[c],0]) &                                                          
                                                (behVec[:,9]==color_pair[cDist[c],1]) &                                                                                                                                              
                                                (behVec[:,10]==6))[0]; 

                        wrong_thisID = np.where((behVec[:,0]==i) &
                                                (behVec[:,2]==nDist[m]) &                
                                                (behVec[:,4]==2) &                                                              
                                                (behVec[:,7]==sDist[s]) &                                                          
                                                (behVec[:,8]==color_pair[cDist[c],0]) &                                                          
                                                (behVec[:,9]==color_pair[cDist[c],1]) &                                                                                                                                              
                                                (behVec[:,10]==6))[0]; 

                        # circle control
                        all_thisID_cc = np.where((behVec[:,0]==i) &
                                                (behVec[:,2]==nDist[m]) &                
                                                (behVec[:,7]==sDist[s]) &                                                          
                                                (behVec[:,8]==color_pair[cDist[c],0]+3) &                                                          
                                                (behVec[:,9]==color_pair[cDist[c],1]) &                                                                                                                                              
                                                (behVec[:,10]==6))[0]; 

                        correct_thisID_cc = np.where((behVec[:,0]==i) &
                                                    (behVec[:,2]==nDist[m]) &                
                                                    (behVec[:,4]==1) &                                                              
                                                    (behVec[:,7]==sDist[s]) &                                                          
                                                    (behVec[:,8]==color_pair[cDist[c],0]+3) &                                                          
                                                    (behVec[:,9]==color_pair[cDist[c],1]) &                                                                                                                                              
                                                    (behVec[:,10]==6))[0]; 

                        wrong_thisID_cc = np.where((behVec[:,0]==i) &
                                                (behVec[:,2]==nDist[m]) &                
                                                (behVec[:,4]==2) &                                                              
                                                (behVec[:,7]==sDist[s]) &                                                          
                                                (behVec[:,8]==color_pair[cDist[c],0]+3) &                                                          
                                                (behVec[:,9]==color_pair[cDist[c],1]) &                                                                                                                                              
                                                (behVec[:,10]==6))[0]; 

                        pCorrect['multiVary'][m,i,s,c] = 100 * (len(correct_thisID) / len(all_thisID)); 
                        correct_RT['multiVary'][m,i,s,c] = np.nanmean(behVec[correct_thisID,5]); 
                        wrong_RT['multiVary'][m,i,s,c] = np.nanmean(behVec[wrong_thisID,5]); 

                        pCorrect['multiVary_cc'][m,i,s,c] = 100 * (len(correct_thisID_cc) / len(all_thisID_cc)); 
                        correct_RT['multiVary_cc'][m,i,s,c] = np.nanmean(behVec[correct_thisID_cc,5]); 
                        wrong_RT['multiVary_cc'][m,i,s,c] = np.nanmean(behVec[wrong_thisID_cc,5]); 


                # separate figure for each size, color
                plt.figure(); 
                plt.gcf().suptitle(f'Size: {sDist[s]}, T.Color: {color_pair[cDist[c],1]}, D.Color: {color_pair[cDist[c],0]}', fontsize=10); 

                for m in np.arange(len(nDist)):
                    # pCorrect
                    plt.subplot(2,len(nDist),m+1); 
                    plt.gca().bar(bar_position, pCorrect['multiVary'][m,:,s,c], facecolor=[0.5,0.5,0.5]);  
                    plt.gca().plot([0, bar_position[int(n_refStim/2)-1]+1.5], 
                                   [np.nanmean(pCorrect['multiVary'][m,:int(n_refStim/2),s,c]), 
                                    np.nanmean(pCorrect['multiVary'][m,:int(n_refStim/2),s,c])],
                                   color='r', linewidth = 2); 
                    plt.gca().plot([bar_position[int(n_refStim/2)]-1.5, bar_position[-1]+1.5], 
                                   [np.nanmean(pCorrect['multiVary'][m,int(n_refStim/2):,s,c]), 
                                    np.nanmean(pCorrect['multiVary'][m,int(n_refStim/2):,s,c])],
                                   color='r', linewidth = 2); 
                    plt.gca().set_xlim([0, bar_position[-1]+1.5]); 
                    plt.gca().set_ylim([0, 100]); 
                    plt.gca().set_xticks(bar_position); 
                    plt.gca().set_xticklabels(xlabels); 
                    plt.gca().set_xlabel('Stimulus ID'); 
                    plt.gca().set_ylabel('%Correct peformance'); 
                    plt.gca().set_title(f'numVary: N = {nDist[m]}'); 

                    # reaction time
                    plt.subplot(2,len(nDist),m+1+len(nDist)); 
                    plt.gca().bar(bar_position_correct, correct_RT['multiVary'][m,:,s,c],width=0.5, color='r', label='correct');
                    plt.gca().bar(bar_position_wrong, wrong_RT['multiVary'][m,:,s,c], width=0.5, color='b', label='wrong'); 
                    plt.gca().plot(0,np.nanmean(correct_RT['multiVary'][m,:int(n_refStim/2),s,c]),'r>', markerfacecolor='r'); 
                    plt.gca().plot(0,np.nanmean(wrong_RT['multiVary'][m,:int(n_refStim/2),s,c]),'b>', markerfacecolor='b'); 
                    plt.gca().plot(bar_position_wrong[-1]+1,np.nanmean(correct_RT['multiVary'][m,int(n_refStim/2):,s,c]),'r<', markerfacecolor='r'); 
                    plt.gca().plot(bar_position_wrong[-1]+1,np.nanmean(wrong_RT['multiVary'][m,int(n_refStim/2):,s,c]),'b<', markerfacecolor='b'); 
                    plt.gca().legend(); 
                    plt.gca().set_xlim([0, bar_position_wrong[-1]+1]); 
                    plt.gca().set_ylim([0, 350]); 
                    plt.gca().set_xticks(bar_position); 
                    plt.gca().set_xticklabels(xlabels); 
                    plt.gca().set_xlabel('Stimulus ID'); 
                    plt.gca().set_ylabel('Reaction time (ms)'); 
                    plt.gca().set_title(f'numVary: N = {nDist[m]}'); 

                plt.tight_layout(); 
                plt.show(); 

                # separate figure for each size, color: Circle control
                plt.figure(); 
                plt.gcf().suptitle(f'Size: {sDist[s]}, T.Color: {color_pair[cDist[c],1]}, D.Color: {color_pair[cDist[c],0]}', fontsize=10); 

                for m in np.arange(len(nDist)):
                    # pCorrect
                    plt.subplot(2,len(nDist),m+1); 
                    plt.gca().bar(bar_position, pCorrect['multiVary_cc'][m,:,s,c], facecolor=[0.5,0.5,0.5]);  
                    plt.gca().plot([0, bar_position[int(n_refStim/2)-1]+1.5], 
                                   [np.nanmean(pCorrect['multiVary_cc'][m,:int(n_refStim/2),s,c]), 
                                    np.nanmean(pCorrect['multiVary_cc'][m,:int(n_refStim/2),s,c])],
                                   color='r', linewidth = 2); 
                    plt.gca().plot([bar_position[int(n_refStim/2)]-1.5, bar_position[-1]+1.5], 
                                   [np.nanmean(pCorrect['multiVary_cc'][m,int(n_refStim/2):,s,c]), 
                                    np.nanmean(pCorrect['multiVary_cc'][m,int(n_refStim/2):,s,c])],
                                   color='r', linewidth = 2); 
                    plt.gca().set_xlim([0, bar_position[-1]+1.5]); 
                    plt.gca().set_ylim([0, 100]); 
                    plt.gca().set_xticks(bar_position); 
                    plt.gca().set_xticklabels(xlabels); 
                    plt.gca().set_xlabel('Stimulus ID'); 
                    plt.gca().set_ylabel('%Correct peformance'); 
                    plt.gca().set_title(f'numVary CC: N = {nDist[m]}'); 

                    # reaction time
                    plt.subplot(2,len(nDist),m+1+len(nDist)); 
                    plt.gca().bar(bar_position_correct, correct_RT['multiVary_cc'][m,:,s,c],width=0.5, color='r', label='correct');
                    plt.gca().bar(bar_position_wrong, wrong_RT['multiVary_cc'][m,:,s,c], width=0.5, color='b', label='wrong'); 
                    plt.gca().plot(0,np.nanmean(correct_RT['multiVary_cc'][m,:int(n_refStim/2),s,c]),'r>', markerfacecolor='r'); 
                    plt.gca().plot(0,np.nanmean(wrong_RT['multiVary_cc'][m,:int(n_refStim/2),s,c]),'b>', markerfacecolor='b'); 
                    plt.gca().plot(bar_position_wrong[-1]+1,np.nanmean(correct_RT['multiVary_cc'][m,int(n_refStim/2):,s,c]),'r<', markerfacecolor='r'); 
                    plt.gca().plot(bar_position_wrong[-1]+1,np.nanmean(wrong_RT['multiVary_cc'][m,int(n_refStim/2):,s,c]),'b<', markerfacecolor='b'); 
                    plt.gca().legend(); 
                    plt.gca().set_xlim([0, bar_position_wrong[-1]+1]); 
                    plt.gca().set_ylim([0, 350]); 
                    plt.gca().set_xticks(bar_position); 
                    plt.gca().set_xticklabels(xlabels); 
                    plt.gca().set_xlabel('Stimulus ID'); 
                    plt.gca().set_ylabel('Reaction time (ms)'); 
                    plt.gca().set_title(f'numVary CC: N = {nDist[m]}'); 

                plt.tight_layout(); 
                plt.show(); 
    print("Behavior data was processed"); 


    ### Neuronal Response
    nClusters = experiment['numNeurons']; 
    print(f'numNeurons: {nClusters}'); 

    #TimeOfInterest = np.arange(int(prevTime*1000+50),int(prevTime*1000+experiment['StimDur']+100+1));
    TimeOfInterest = np.arange(int(prevTime*1000+50),int(prevTime*1000+300+1));
    print(f"TimeOfInterest: {len(TimeOfInterest)} after stim onset"); 

    StimResp = []; 
    mResp = np.zeros((numTrials,nClusters)); 
    psth_mtx = np.zeros((numTrials,int(300 + prevTime*1000*2),nClusters));     
    psth_mtx[:] = np.nan;     
    for i in np.arange(numTrials):
        StimResp.append(dict());
        StimResp[i]['timeOn'] = experiment['stimStructs'][i]['timeOn']; 
        StimResp[i]['timeOff'] = experiment['stimStructs'][i]['timeOff'];    
        StimResp[i]['pdOn'] = experiment['stimStructs'][i]['pdOn']; 
        StimResp[i]['pdOff'] = experiment['stimStructs'][i]['pdOff'];        
        StimResp[i]['neurons'] = experiment['stimStructs'][i]['neurons'];        
        #print(f"timeOn = {StimResp[i]['timeOn']}");
        #print(f"timeOff = {StimResp[i]['timeOff']}");
        #print(f"pdOn = {StimResp[i]['pdOn']}");
        #print(f"pdOff = {StimResp[i]['pdOff']}");

        for j in np.arange(nClusters):
            #sigLength = int(experiment['StimDur'] + prevTime*1000*2); # to include pre_stim, post_stim
            sigLength = int(300 + prevTime*1000*2); # to include pre_stim, post_stim
            StimResp[i]['neurons'][j]['spkMtx'] = np.zeros((1,sigLength),dtype=int);
            
            if 'spikes' in StimResp[i]['neurons'][j].keys():
                spkTime = StimResp[i]['neurons'][j]['spikes'] - StimResp[i]['pdOn'];
                spkTime = spkTime[:]*1000 + prevTime*1000;
                spkTime = spkTime.astype(int);
                spkTime = spkTime[np.where(spkTime<sigLength)];
                StimResp[i]['neurons'][j]['spkMtx'][0,spkTime] = 1;
                
                StimResp[i]['neurons'][j]['meanSDF'] = makeSDF.getSDF(StimResp[i]['neurons'][j]['spkMtx'],1000);
                mResp[i,j] = np.mean(StimResp[i]['neurons'][j]['meanSDF'][TimeOfInterest])
                psth_mtx[i,:,j] = makeSDF.getSDF(StimResp[i]['neurons'][j]['spkMtx'],1000);     
            else:
                mResp[i,j] = np.nan; 
        print(f"StimResp {i}/{numTrials} was processed"); 
    

    del experiment['stimStructs']; 
    del experiment['iti_start']; 
    del experiment['iti_end']; 
    experiment['filename'] = dat_filename; 
    experiment['StimResp'] = StimResp; 
    experiment['behVec'] = behVec; 
    #experiment['pCorrect'] = pCorrect;    
    #experiment['correct_RT'] = correct_RT;    
    #experiment['wrong_RT'] = wrong_RT;        

    ### save experiment (processed file)
    path_to_save = imec_filename[:(imec_filename.rfind('/')+1)] + 'processed/'; 
    if os.path.exists(path_to_save)==0:
        os.mkdir(path_to_save); 
    name_to_save = path_to_save + bin_filename[(bin_filename.rfind('/')+1):-8] + 'npz';
    #name_to_save = path_to_save + bin_filename[(bin_filename.rfind('\\')+1):-8] + 'npz';
    np.savez_compressed(name_to_save, **experiment); 

    #name_to_save = path_to_save + bin_filename[(bin_filename.rfind('/')+1):-8] + 'json.gz';
    #name_to_save = path_to_save + bin_filename[(bin_filename.rfind('\\')+1):-8] + 'json.gz';    
    #f = gzip.GzipFile(name_to_save,'wb');    
    #f.write(json.dumps(experiment, cls=NumpyEncoder).encode('utf-8')); 
    #f.close(); 
    print('processed file was saved'); 


    ### Drawing Part ###
    plt.figure(figsize=(6, 3)); 
    plt.plot(np.arange(-prevTime*1000,prevTime*1000+300),
             np.nanmean(np.nanmean(psth_mtx,axis=0),axis=1)); 

    # sort neurons according to response level (strong to weak)
    neurons_from_strong = np.nansum(mResp,axis=0).argsort()[::-1]; 

    # target alone
    tAlone = []; 
    tAlone_mtx = np.zeros((n_refStim,nClusters));   # row: stims, col: cells
    for i in np.arange(n_refStim):
        tAlone.append(dict()); 
        selected = np.where((behVec[:,0]==i) & (behVec[:,2]==0))[0]; 
        tAlone[i]['trials'] = selected; 
        tAlone[i]['neurons'] = []; 
        for j in np.arange(nClusters):
            tAlone[i]['neurons'].append(dict()); 
            tAlone[i]['neurons'][j]['SDF'] = np.zeros((len(selected),sigLength)); 
            for r in np.arange(len(selected)):
                tAlone[i]['neurons'][j]['SDF'][r,:] = StimResp[selected[r]]['neurons'][j]['meanSDF']; 
            meanSDF_now = np.mean(tAlone[i]['neurons'][j]['SDF'],axis=0);     
            tAlone_mtx[i,j] = np.mean(meanSDF_now[TimeOfInterest]); 

    # numVary
    numVary = []; 
    selected = np.where(behVec[:,10]==1)[0]; 
    nDist = np.unique(behVec[selected,2]); 
    numVary_mtx = np.zeros((n_refStim,nClusters,len(nDist)));   # stimID x cells x nDist   
    for m in np.arange(len(nDist)):
        numVary.append(dict()); 
        numVary[m]['numDist'] = nDist[m]; 
        numVary[m]['target'] = []; 
        for i in np.arange(n_refStim):
            numVary[m]['target'].append(dict()); 
            trials_now = np.where((behVec[:,0]==i) &
                                  (behVec[:,2]==nDist[m]) &                
                                  (behVec[:,8]==0) &                                                      
                                  (behVec[:,10]==1))[0]; 
            numVary[m]['target'][i]['trials'] = trials_now; 
            numVary[m]['target'][i]['neurons'] = []; 
            for j in np.arange(nClusters):
                numVary[m]['target'][i]['neurons'].append(dict()); 
                numVary[m]['target'][i]['neurons'][j]['SDF'] = np.zeros((len(trials_now),sigLength)); 
                for r in np.arange(len(trials_now)):
                    numVary[m]['target'][i]['neurons'][j]['SDF'][r,:] =  StimResp[trials_now[r]]['neurons'][j]['meanSDF']; 
                meanSDF_now = np.mean(numVary[m]['target'][i]['neurons'][j]['SDF'],axis=0);     
                numVary_mtx[i,j,m] = np.mean(meanSDF_now[TimeOfInterest]); 

    # numVary_cc
    numVary_cc = []; 
    numVary_cc_mtx = np.zeros((n_refStim,nClusters,len(nDist)));   # stimID x cells x nDist   
    for m in np.arange(len(nDist)):
        numVary_cc.append(dict()); 
        numVary_cc[m]['numDist'] = nDist[m]; 
        numVary_cc[m]['target'] = []; 
        for i in np.arange(n_refStim):
            numVary_cc[m]['target'].append(dict()); 
            trials_now = np.where((behVec[:,0]==i) &
                                  (behVec[:,2]==nDist[m]) &                
                                  (behVec[:,8]==3) &                                                      
                                  (behVec[:,10]==1))[0]; 
            numVary_cc[m]['target'][i]['trials'] = trials_now; 
            numVary_cc[m]['target'][i]['neurons'] = [];             
            for j in np.arange(nClusters):
                numVary_cc[m]['target'][i]['neurons'].append(dict()); 
                numVary_cc[m]['target'][i]['neurons'][j]['SDF'] = np.zeros((len(trials_now),sigLength)); 
                for r in np.arange(len(trials_now)):
                    numVary_cc[m]['target'][i]['neurons'][j]['SDF'][r,:] =  StimResp[trials_now[r]]['neurons'][j]['meanSDF']; 
                meanSDF_now = np.mean(numVary_cc[m]['target'][i]['neurons'][j]['SDF'],axis=0);     
                numVary_cc_mtx[i,j,m] = np.mean(meanSDF_now[TimeOfInterest]); 

    # draw figure from strong units
    plt.figure(figsize=(6, 6))
    ax1 = plt.subplot(2,2,1); 
    ax2 = plt.subplot(2,2,2); 
    ax3 = plt.subplot(2,1,2);     
    for j in np.arange(nClusters):
        unit_now = neurons_from_strong[j];             
        if app.all_radiobutton.isChecked() == True:
            unit_id = experiment['neuronid'][unit_now]; 
        elif app.sua_radiobutton.isChecked() == True:
            unit_id = experiment['id_sua'][unit_now]; 
        elif app.mua_radiobutton.isChecked() == True:
            unit_id = experiment['id_mua'][unit_now]; 

        target_preference = tAlone_mtx[:,unit_now].argsort()[::-1]; 

        ax1.clear(); 
        ax1.plot(tAlone_mtx[target_preference,unit_now],'ro-'); 
        for m in np.arange(len(nDist)):        
            ax1.plot(numVary_mtx[target_preference,unit_now,m],'o-',
                     color=(1-0.2*m,1-0.2*m,1-0.2*m),
                     markerfacecolor=(1-0.2*m,1-0.2*m,1-0.2*m),
                     label=f'nDist = {nDist[m]}'); 
        ax1.set_xlabel('Stim ID (from strong to weak)'); 
        ax1.set_ylabel('Responses (spikes/s)'); 
        ax1.set_title('NumVary'); 
        ax1.spines.right.set_visible(False); 
        ax1.spines.top.set_visible(False);         

        ax2.clear(); 
        ax2.plot(tAlone_mtx[target_preference,unit_now],'ro-'); 
        for m in np.arange(len(nDist)):        
            ax2.plot(numVary_cc_mtx[target_preference,unit_now,m],'o:',
                     color=(1-0.2*m,1-0.2*m,1-0.2*m),
                     markerfacecolor=(1-0.2*m,1-0.2*m,1-0.2*m),
                     label=f'nDist = {nDist[m]}'); 
        ax2.set_xlabel('Stim ID (from strong to weak)'); 
        ax2.set_ylabel('Responses (spikes/s)'); 
        ax2.set_title('NumVary CC'); 
        ax2.spines.right.set_visible(False); 
        ax2.spines.top.set_visible(False);         

        ax3.clear(); 
        meanSDF_mtx = []; 
        for i in np.arange(n_refStim):
            target_id = target_preference[i];
            meanSDF_now = np.mean(tAlone[target_id]['neurons'][unit_now]['SDF'],axis=0); 
            color_now = np.array([1,0,0]) + np.array([0,1/n_refStim,1/n_refStim])*i; 
            ax3.plot(np.arange(-prevTime*1000,prevTime*1000+300),
                     meanSDF_now, color=color_now); 
            meanSDF_mtx.append(meanSDF_now); 
        meanSDF_mtx.append(meanSDF_now); 
        ax3.plot(np.arange(-prevTime*1000,prevTime*1000+300),
                 np.mean(meanSDF_mtx,axis=0), color=[0,0,0]); 
        ax3.set_xlabel('Time from stimulus onset (ms)'); 
        ax3.set_ylabel('Response (spikes/s)'); 
        ax3.set_title(f'Target alone: j = {j}/{nClusters}: unit_id = {unit_id}'); 
        ax3.spines.right.set_visible(False); 
        ax3.spines.top.set_visible(False);         

        print(f'j = {j}/{nClusters}: unit_id = {unit_id}'); 
        plt.tight_layout();       
        plt.pause(3); 

        if app.running == 0:
            break; 
    plt.show(); 



class NumpyEncoder(json.JSONEncoder):
    # Special json encoder for numpy types 

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj,np.ndarray): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


"""
from scipy.io import savemat
import numpy as np
import glob
import os
npzFiles = ["/Users/taekjunkim/Downloads/x230517_beh_g1_t0.npz"];

for f in npzFiles:
    fm = os.path.splitext(f)[0]+'.mat'
    d = np.load(f, allow_pickle=True)
    savemat(fm, d)
    print('generated ', fm, 'from', f)
"""
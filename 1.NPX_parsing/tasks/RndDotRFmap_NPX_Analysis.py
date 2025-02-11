#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:38:14 2019

RndDotRFmap_Analysis.py

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
from helper import makeSDF;
from helper import parseTJexperiment_NPX as parse_NPX;
import json;
import gzip;




#%%
def main(app):
    #%% This needs to be defined
    dat_filename = app.dat_file; 
    imec_filename = app.imec_file; 
    task_folder = dat_filename[:(dat_filename.rfind('/')+1)]; 
    bin_filename = glob.glob(task_folder+'*.nidq.bin')[0]; 


    prevTime = 0.2; 
    markervals, markervals_str = parse_NPX.get_markervals(dat_filename); 

    numStims = (int(markervals_str[5])-int(markervals_str[1])+1) * (int(markervals_str[7])-int(markervals_str[3])+1) + 1; 

    experiment = parse_NPX.main(bin_filename, dat_filename, prevTime, numStims, imec_filename, app); 

    xRange = [experiment['RangeXmin'], experiment['RangeXmax']]; 
    yRange = [experiment['RangeYmin'], experiment['RangeYmax']]; 
    experiment['StimDur'] = experiment['stimon']; 

    experiment['RF_fit'] = []; 
    for i in np.arange(experiment['numNeurons']):
        experiment['RF_fit'].append(dict()); 
        experiment['RF_fit'][i]['RFmap'] = np.nan; 
        experiment['RF_fit'][i]['amplitude'] = np.nan;         
        experiment['RF_fit'][i]['x0'] = np.nan; 
        experiment['RF_fit'][i]['y0'] = np.nan;  
        experiment['RF_fit'][i]['sigma_x'] = np.nan; 
        experiment['RF_fit'][i]['sigma_y'] = np.nan;
        experiment['RF_fit'][i]['theta'] = np.nan;   
        experiment['RF_fit'][i]['offset'] = np.nan;   
        experiment['RF_fit'][i]['fit_r'] = np.nan;   

    TimeOfInterest = np.arange(int(experiment['isi']+50),int(experiment['isi']+experiment['StimDur']+100+1));

    #%%
    StimResp = []; 
    mResp = np.zeros((numStims,experiment['numNeurons'])); 
    psth_mtx = np.zeros((numStims,int(experiment['StimDur'] + experiment['isi']*2),experiment['numNeurons'])); 
    for i in np.arange(len(experiment['stimStructs'])):
        StimResp.append(dict());
        StimResp[i]['timeOn'] = experiment['stimStructs'][i]['timeOn']; 
        StimResp[i]['timeOff'] = experiment['stimStructs'][i]['timeOff'];    
        StimResp[i]['pdOn'] = experiment['stimStructs'][i]['pdOn']; 
        StimResp[i]['pdOff'] = experiment['stimStructs'][i]['pdOff'];        
        StimResp[i]['neurons'] = experiment['stimStructs'][i]['neurons'];        

        if i<numStims-1:
            StimResp[i]['xPos'] = int(np.floor(i/(xRange[-1]-xRange[0]+1))) + xRange[0];
            StimResp[i]['yPos'] = i%(xRange[-1]-xRange[0]+1) + yRange[0];    
        else:
            StimResp[i]['xPos'] = np.nan;
            StimResp[i]['yPos'] = np.nan;        


        for j in np.arange(experiment['numNeurons']):
            NumRepeat = len(StimResp[i]['pdOn']);
            sigLength = int(experiment['StimDur'] + experiment['isi']*2); # to include pre_stim, post_stim
            StimResp[i]['neurons'][j]['spkMtx'] = np.zeros((NumRepeat,sigLength),dtype=int);
            
            for r in np.arange(NumRepeat):
                spkTime = StimResp[i]['neurons'][j]['spikes'][r] - StimResp[i]['pdOn'][r];
                spkTime = spkTime[:]*1000 + experiment['isi'];
                spkTime = spkTime.astype(int);
                spkTime = spkTime[np.where(spkTime<sigLength)];

                StimResp[i]['neurons'][j]['spkMtx'][r,spkTime] = 1;
            
            StimResp[i]['neurons'][j]['meanSDF'] = makeSDF.getSDF(StimResp[i]['neurons'][j]['spkMtx'],1000);
            psth_mtx[i,:,j] = makeSDF.getSDF(StimResp[i]['neurons'][j]['spkMtx'],1000);
            mResp[i,j] = np.mean(StimResp[i]['neurons'][j]['meanSDF'][TimeOfInterest])

    experiment['mResp'] = mResp; 

    #%%
    x = np.arange(xRange[0],xRange[1]+1,1);
    y = np.arange(yRange[0],yRange[1]+1,1);
    x, y = np.meshgrid(x,y);
    posData = np.vstack((x.ravel(),y.ravel()));


    x2 = np.arange(xRange[0],xRange[1]+0.01,0.01);
    y2 = np.arange(yRange[0],yRange[1]+0.01,0.01);
    x2, y2 = np.meshgrid(x2,y2);
    posData2 = np.vstack((x2.ravel(),y2.ravel())); 

    # sort neurons according to response level (strong to weak)
    neurons_from_strong = np.sum(mResp,axis=0).argsort()[::-1]; 

    plt.figure(figsize=(9, 9))
    ax1 = plt.subplot(3,3,1); 
    ax2 = plt.subplot(3,3,2); 
    ax3 = plt.subplot(3,3,3);     
    ax4 = plt.subplot(3,3,4); 
    ax5 = plt.subplot(3,3,5); 
    ax7 = plt.subplot(3,3,7);                     
    ax8 = plt.subplot(3,3,8);                     


    nClusters = experiment['numNeurons']; 
    print(f'numNeurons: {nClusters}'); 

    # electrode channel depth range
    depth_min = np.min(np.array(experiment['chpos_sua']+experiment['chpos_mua'])[:,1]); 
    depth_max = np.max(np.array(experiment['chpos_sua']+experiment['chpos_mua'])[:,1]); 
    cmap = plt.cm.viridis; 

    num_sua = 0; 
    num_mua = 0; 

    for j in np.arange(experiment['numNeurons']):

        RFmap = np.zeros((yRange[1]-yRange[0]+1,xRange[1]-xRange[0]+1));

        unit_now = neurons_from_strong[j]; 
        if app.all_radiobutton.isChecked() == True:
            unit_id = experiment['neuronid'][unit_now]; 
        elif app.sua_radiobutton.isChecked() == True:
            unit_id = experiment['id_sua'][unit_now]; 
        elif app.mua_radiobutton.isChecked() == True:
            unit_id = experiment['id_mua'][unit_now]; 

        for i in np.arange(numStims):
            if i<numStims-1:
                rowNum = i%np.shape(x)[0];
                colNum = int(np.floor(i/np.shape(x)[0]));
                RFmap[rowNum,colNum] = mResp[i,unit_now];

        #RFmap = RFmap - mResp[-1,j];        
        RFmap[RFmap[:]<0] = 0.01;    
        RFmap_norm = RFmap/np.max(abs(RFmap));
        spon = mResp[-1,j]; 
        spon_norm = spon/np.max(abs(RFmap)); 

        maxPos = np.where(RFmap==np.max(RFmap)); 
        if len(maxPos[0])>0:
            maxPos = [maxPos[0][0],maxPos[1][0]]; 
        else:
            maxPos = [xRange[0]+3, yRange[0]+3]; 
        #if len(maxPos[0])>1:
        #    maxPos = maxPos[0]

        # to avoid maximum value on boundary
        maxPos = list(maxPos)
        for xy in range(2):
            if maxPos[xy]==0:
                maxPos[xy] += 1
            elif maxPos[xy]==6:
                maxPos[xy] -= 1

        experiment['RF_fit'][unit_now]['RFmap'] = RFmap; 

        # fitting parameter
        # amplitude, x0, y0, sigma_x, sigma_y, theta, offset

        init_guess = [1, xRange[0]+int(maxPos[1]), yRange[0]+int(maxPos[0]),
                      2, 2, 0.5*np.pi, spon_norm]; 
        param_bounds = [(0, xRange[0], yRange[0], 0.5, 0.5, 0, 0),
                        (2, xRange[1], yRange[1], 5, 5, 1*np.pi, 1)]

        """
        init_guess = [1,xRange[0]+2,yRange[0]+2,3,3,0,0.1]; #amp,xo,yo,sigx,sigy,theta,offset
        param_bounds = [(0, xRange[0], yRange[0], 0.5, 0.5, 0, 0),
                        (2, xRange[1], yRange[1], 5, 5, 1*np.pi, 1)]
        """


        try:
            popt, pcov = opt.curve_fit(twoD_Gaussian, posData, RFmap_norm.ravel(), 
                                       p0=init_guess, bounds=param_bounds); 
            data_fitted0 = twoD_Gaussian(posData, *popt);     
            fit_r = np.corrcoef(data_fitted0, RFmap_norm.ravel())[0,1]; 

            if fit_r>np.sqrt(0.49):
                data_fitted = twoD_Gaussian(posData2, *popt); 

                if unit_id in experiment['id_sua']:
                    contour_color = [0, 0, 0]; 
                    sua_idx = np.where(experiment['id_sua']==unit_id)[0][0]; 
                    ch_xc = experiment['chpos_sua'][sua_idx][0]; 
                    ch_yc = experiment['chpos_sua'][sua_idx][1]; 

                elif unit_id in experiment['id_mua']:                    
                    contour_color = [0.5, 0.5, 0.5];                     
                    mua_idx = np.where(experiment['id_mua']==unit_id)[0][0];                     
                    ch_xc = experiment['chpos_mua'][mua_idx][0]; 
                    ch_yc = experiment['chpos_mua'][mua_idx][1]; 


                ax1.contour(x2, y2, (data_fitted.reshape(np.shape(x2))-popt[-1])/popt[0], 
                            [0.5], colors = [contour_color]); 
                ax1.set_xlim(xRange[0]-0.5,xRange[1]+0.5);
                ax1.set_ylim(yRange[0]-0.5,yRange[1]+0.5);    
                ax1.set_title('RF fitted');    
                ax1.set_xlabel('RF horizontal (deg)'); 
                ax1.set_ylabel('RF vertical (deg)')

                ax2.imshow(RFmap,origin='lower',extent=(xRange[0]-0.5,xRange[1]+0.5,
                                                     yRange[0]-0.5,yRange[1]+0.5));      
                ax2.set_title(f'RF map: fit_r = {round(fit_r,3)}')     
                ax2.set_xlabel('RF horizontal (deg)'); 
                ax2.set_ylabel('RF vertical (deg)')

                ax3.clear(); 
                ax3.plot(np.mean(psth_mtx[:,:,unit_now],axis=0)); 
                ax3.set_xticks(np.arange(0,np.shape(psth_mtx)[1],experiment['isi'])); 
                ax3.set_xticklabels(np.arange(0,np.shape(psth_mtx)[1],experiment['isi'])-experiment['isi']); 
                ax3.set_xlabel('Time from stimulus onset (ms)'); 
                ax3.set_ylabel('Response (Hz)')
                
                dot_color = cmap((ch_yc-depth_min)/(depth_max-depth_min)); 

                if unit_id in experiment['id_sua']:
                    num_sua += 1; 
                    ax4.plot(popt[1], ch_yc, 'o', c = dot_color);  
                    ax4.set_xlim(xRange[0]-0.5,xRange[1]+0.5);
                    ax4.set_title(f'SUA (n = {num_sua})', fontweight='bold'); 
                    ax4.set_xlabel('RF horizontal (deg)'); 
                    ax4.set_ylabel('Distance from the tip (micrometer)')

                    ax7.plot(popt[2], ch_yc, 'o', c = dot_color);  
                    ax7.set_xlim(yRange[0]-0.5,yRange[1]+0.5);    
                    ax7.set_xlabel('RF vertical (deg)'); 
                    ax7.set_ylabel('Distance from the tip (micrometer)')


                elif unit_id in experiment['id_mua']:                    
                    num_mua += 1;                     
                    ax5.plot(popt[1], ch_yc, 'o', c = dot_color);  
                    ax5.set_xlim(xRange[0]-0.5,xRange[1]+0.5);
                    ax5.set_title(f'MUA (n = {num_mua})', fontweight='bold'); 
                    ax5.set_xlabel('RF horizontal (deg)'); 
                    ax5.set_ylabel('Distance from the tip (micrometer)')

                    ax8.plot(popt[2], ch_yc, 'o', c = dot_color);  
                    ax8.set_xlim(yRange[0]-0.5,yRange[1]+0.5);    
                    ax8.set_xlabel('RF vertical (deg)'); 
                    ax8.set_ylabel('Distance from the tip (micrometer)')
            else:
                ax2.imshow(RFmap,origin='lower',extent=(xRange[0]-0.5,xRange[1]+0.5,
                                                     yRange[0]-0.5,yRange[1]+0.5));      
                ax2.set_title(f'RF map: fit_r = {round(fit_r,3)}')     
                ax2.set_xlabel('RF horizontal (deg)'); 
                ax2.set_ylabel('RF vertical (deg)')

                ax3.clear(); 
                ax3.plot(np.mean(psth_mtx[:,:,unit_now],axis=0)); 
                ax3.set_xticks(np.arange(0,np.shape(psth_mtx)[1],experiment['isi'])); 
                ax3.set_xticklabels(np.arange(0,np.shape(psth_mtx)[1],experiment['isi'])-experiment['isi']); 
                ax3.set_xlabel('Time from stimulus onset (ms)'); 
                ax3.set_ylabel('Response (Hz)')

                pass; 


            experiment['RF_fit'][unit_now]['amplitude'] = popt[0]; 
            experiment['RF_fit'][unit_now]['x0'] = popt[1]; 
            experiment['RF_fit'][unit_now]['y0'] = popt[2];  
            if popt[3] >= popt[4]:
                experiment['RF_fit'][unit_now]['sigma_x'] = popt[3]; 
                experiment['RF_fit'][unit_now]['sigma_y'] = popt[4];
                experiment['RF_fit'][unit_now]['theta'] = popt[5];   
            else:
                experiment['RF_fit'][unit_now]['sigma_x'] = popt[4]; 
                experiment['RF_fit'][unit_now]['sigma_y'] = popt[3];  
                if popt[5] > np.pi/2:
                    experiment['RF_fit'][unit_now]['theta'] = popt[5]-np.pi/2;
                else:                       
                    experiment['RF_fit'][unit_now]['theta'] = popt[5]+np.pi/2;                    
            experiment['RF_fit'][unit_now]['offset'] = popt[6];   
            experiment['RF_fit'][unit_now]['fit_r'] = fit_r;   
                
        except Exception as e:
            print(e); 
            print(f'No fit found for unit#: {unit_now}'); 

        print(f'j = {j}/{nClusters}: unit_id = {unit_id}'); 
        plt.tight_layout();       
        plt.pause(0.5); 

        if app.running == 0:
            break; 

    """
    #for j in np.arange(experiment['numNeurons']):         
    for j in np.arange(6):             
        plt.subplot(4,3,j+1);
        RFmap = np.zeros((xRange[1]-xRange[0]+1,yRange[1]-yRange[0]+1));

        unit_now = neurons_from_strong[j]; 
        for i in np.arange(numStims):
            if i<numStims-1:
                rowNum = i%len(x);
                colNum = int(np.floor(i/len(x)));
                RFmap[rowNum,colNum] = mResp[i,unit_now];
        #RFmap = RFmap - mResp[-1,j];        
        RFmap[RFmap[:]<0] = 0.01;    
        RFmap = RFmap/np.max(abs(RFmap));
        #plt.imshow(RFmap,vmin=-1,vmax=1,cmap='bwr',origin='lower');  
        plt.imshow(RFmap,origin='lower',extent=(xRange[0]-0.5,xRange[1]+0.5,
                                                yRange[0]-0.5,yRange[1]+0.5));      
        cluster_id = experiment['neuronid'][unit_now]; 
        plt.title(f'unit#: {unit_now}, cluster_id: {cluster_id}');                                                

        plt.subplot(4,3,j+1+6);
        popt, pcov = opt.curve_fit(twoD_Gaussian, posData, RFmap.ravel(), p0=init_guess)
        data_fitted = twoD_Gaussian(posData2, *popt);
        plt.imshow(data_fitted.reshape(np.shape(x2)),origin='lower',extent=(xRange[0],xRange[1],
                                                yRange[0],yRange[1]));         
        plt.contour(x2, y2, (data_fitted.reshape(np.shape(x2))-popt[-1])/popt[0], [0.5], colors='w')
        plt.xlim(xRange[0]-0.5,xRange[1]+0.5);
        plt.ylim(yRange[0]-0.5,yRange[1]+0.5);    

        cluster_id = experiment['neuronid'][unit_now]; 
        plt.title(f'unit#: {unit_now}, cluster_id: {cluster_id}');                                                
    """
    plt.show(block=True);

#%%
    del experiment['stimStructs'];
    del experiment['iti_start'];
    del experiment['iti_end'];
    experiment['filename'] = dat_filename;
    experiment['StimResp'] = StimResp;
    experiment['xRange'] = xRange;
    experiment['yRange'] = yRange;

    ### save experiment (processed file)
    path_to_save = imec_filename[:(imec_filename.rfind('/')+1)] + 'processed/'; 
    if os.path.exists(path_to_save)==0:
        os.mkdir(path_to_save); 
    name_to_save = path_to_save + bin_filename[(bin_filename.rfind('/')+1):-8] + 'npz';
    np.savez_compressed(name_to_save, **experiment); 

    #name_to_save = path_to_save + bin_filename[(bin_filename.rfind('/')+1):-8] + 'json.gz';
    #f = gzip.GzipFile(name_to_save,'w');    
    #f.write(json.dumps(experiment, cls=NumpyEncoder).encode('utf-8')); 
    #f.close(); 
    print('processed file was saved'); 


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

def twoD_Gaussian(posData, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x, y) = posData;
    xo = float(xo);
    yo = float(yo);         
    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2);
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2);
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2);
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)));

    """
    # constraint to make sigma_x > sigma_y 
    # this is critically associated with theta
    if sigma_y > sigma_x: 
        penalty = (sigma_y - sigma_x) * 100; 
    else:
        penalty = 0; 

    g = g + penalty; 
    """

    return g.ravel()


#%%
"""
depth_min = np.min([np.min(data['chpos_sua'][:,1]),np.min(data['chpos_mua'][:,1])]); 
depth_max = np.max([np.max(data['chpos_sua'][:,1]),np.max(data['chpos_mua'][:,1])]);  
cmap = plt.cm.viridis; 

plt.figure(figsize=(9, 3))
ax1 = plt.subplot(1,3,1); 
ax1.set_xlim(xRange[0]-0.5,xRange[1]+0.5);
ax1.set_ylim(yRange[0]-0.5,yRange[1]+0.5);    
ax1.set_title('RF fitted');    
ax1.set_xlabel('RF horizontal (deg)'); 
ax1.set_ylabel('RF vertical (deg)')
ax1.grid()
ax2 = plt.subplot(1,3,2); 
ax2.set_xlim(xRange[0]-0.5,xRange[1]+0.5);
ax2.set_xlabel('RF horizontal (deg)'); 
ax2.set_ylabel('Distance from the tip (micrometer)')
ax2.spines[['right', 'top']].set_visible(False)
ax3 = plt.subplot(1,3,3);  
ax3.set_xlim(yRange[0]-0.5,yRange[1]+0.5);    
ax3.set_xlabel('RF vertical (deg)'); 
ax3.set_ylabel('Distance from the tip (micrometer)')
ax3.spines[['right', 'top']].set_visible(False)


for i in range(len(data['neuronid'])):

    if data['RF_fit'][i]['fit_r']>0.9:
        popt = [data['RF_fit'][i]['amplitude'],
                data['RF_fit'][i]['x0'], 
                data['RF_fit'][i]['y0'],
                data['RF_fit'][i]['sigma_x'],
                data['RF_fit'][i]['sigma_y'],
                data['RF_fit'][i]['theta'],
                data['RF_fit'][i]['offset']]; 
        data_fitted = twoD_Gaussian(posData2, *popt);                 
        
        unit_id = data['neuronid'][i]; 
        if unit_id in data['id_sua']:
            sua_idx = np.where(data['id_sua']==unit_id)[0][0]; 
            ch_xc = data['chpos_sua'][sua_idx][0]; 
            ch_yc = data['chpos_sua'][sua_idx][1]; 
        elif unit_id in data['id_mua']:                    
            mua_idx = np.where(data['id_mua']==unit_id)[0][0];                     
            ch_xc = data['chpos_mua'][mua_idx][0]; 
            ch_yc = data['chpos_mua'][mua_idx][1]; 
        dot_color = cmap((ch_yc-depth_min)/(depth_max-depth_min)); 

        ax1.contour(x2, y2, (data_fitted.reshape(np.shape(x2))-popt[-1])/popt[0], 
                    [0.5], colors = [dot_color]);
        ax2.plot(popt[1], ch_yc, 'o', c = dot_color);  
        ax3.plot(popt[2], ch_yc, 'o', c = dot_color);  
plt.tight_layout()
plt.savefig('/Volumes/TK_exHDD1/NPX/V4_clutter/x230518_combined_g0/processed/RFmap_x230518.pdf')
"""
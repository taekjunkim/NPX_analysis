"""
I want to see whether shape selectivity in V2 is well explained by orientation tuning
"""

#%% Import modules
import numpy as np; 
import h5py; 
import os.path
import json;
import gzip;

# NWB related
from pynwb import NWBHDF5IO                        # to read/write
from pynwb import NWBFile                          # to set up
from pynwb.ecephys import LFP, ElectricalSeries    # to add raw data
from pynwb.file import Subject

# datetime related
from datetime import datetime
from dateutil import tz

import glob

import pandas as pd; 

from scipy import stats; 
import statsmodels.api as sm; 
from sklearn.linear_model import LinearRegression; 

#%% NWB SET UP

# Year, Month, Day, Hour, Minute, Second
start_time = datetime(2023, 9, 12, 10, 30, 0, tzinfo=tz.gettz('US/Pacific')); 

### general information
nwbfile = NWBFile(
    identifier='Anesthetized_V2',
    session_description='Position Invariance test with Shape stimuli. Anesthetized V2',  # required
    session_start_time=start_time,  # required
    experimenter='Taekjun Kim',  # optional
    institution='University of Washington',  # optional
    # related_publications='DOI:10.1016/j.neuron.2016.12.011',  # optional    
    lab='Pasupathy lab'   # optional
)

### trial columns
nwbfile.add_trial_column(name="session_id", description="session id")
nwbfile.add_trial_column(name="session_name", description="session name")
nwbfile.add_trial_column(name="cond_num", description="zero-based condition number")
nwbfile.add_trial_column(name="shape_id", description="zero-based shape id (0~369)")
nwbfile.add_trial_column(name="pos_id", description="zero-based position (0~4)")

### unit columns
nwbfile.add_unit_column(name="session_id", description="session id")
nwbfile.add_unit_column(name="unit_id", description="neuron id")
nwbfile.add_unit_column(name="depth", description="distance from the electrode tip")
nwbfile.add_unit_column(name="psth", description="psth")
nwbfile.add_unit_column(name="best_pos", description="best position id")
nwbfile.add_unit_column(name="reg_beta", description="coefficient of linear regression")
nwbfile.add_unit_column(name="fit_r", description="regression fit: r")
nwbfile.add_unit_column(name="fit_fp", description="regression fit: f_pval")


#%% Check files
gzip_path = '/Volumes/TK_exHDD2/AnesthetizedV2/ShapePositionInvariance/'; 
table_path = gzip_path + 'stimTable/'; 

gzip_files = glob.glob(gzip_path+'*.json.gz'); 
gzip_files.sort(); 

table_files = glob.glob(table_path+'*.csv'); 
table_files.sort(); 


#%% Load data, process units
stimDict = get_stimDict(); 

for i in range(len(gzip_files)):
    gzip_now = gzip_files[i]; 
    table_now = table_files[i]; 

    # load processed data
    with gzip.GzipFile(gzip_now,'r') as f: 
        Data = json.loads(f.read().decode('utf-8')); 
    print(f'data#{i} was loaded'); 

    # load table data
    df = pd.read_csv(table_now); 

    # update stim condition in trial table of nwb
    for id in range(len(df)):
        nwbfile.add_trial(
            start_time = -0.1,
            stop_time = 0.5,
            session_id = i,
            session_name = gzip_now.rsplit('/')[-1], 
            cond_num = id,
            shape_id = df['shape_id'][id], 
            pos_id = df['pos_id'][id], 
        )
    shape_tested = df[df['pos_id']==0]['shape_id'].values.astype(int); 
    # for linear regression
    xdata = stats.zscore(np.array(stimDict['resp4ori'])[shape_tested,:]); 
    print(f'data#{i}: stim information added'); 

    # update units
    numNeurons = Data['numNeurons']; 
    for j in range(numNeurons):
        psth = []; 
        for id in range(len(df)):
            psth_now = Data['StimResp'][id]['neurons'][j]['meanSDF'][200:800]; 
            psth.append(psth_now); 
        psth = np.array(psth); 

        mResp = np.mean(psth[:,100:500],axis=1); 
        posResp = [np.nanmean(mResp[np.arange(0,250,5)]), 
                   np.nanmean(mResp[np.arange(1,250,5)]), 
                   np.nanmean(mResp[np.arange(2,250,5)]),
                   np.nanmean(mResp[np.arange(3,250,5)]),
                   np.nanmean(mResp[np.arange(4,250,5)])]; 
        bestPos = np.min(np.where(posResp==np.max(posResp))[0]); 
        best_resp = mResp[np.arange(bestPos,250,5)]; 

        # for linear regression
        ydata = stats.zscore(best_resp); 
        est = sm.OLS(ydata, xdata); 

        nwbfile.add_unit(
            session_id = i,
            unit_id = Data['neuronid'][j], 
            depth = Data['chpos_sua'][j],
            psth = psth, 
            best_pos = bestPos,
            reg_beta = est.fit().params,
            fit_r = est.fit().rsquared**0.5,
            fit_fp = est.fit().f_pvalue,
        )
    print(f'data#{i}: unit information added'); 


nwb_filename = 'ori_regression_DB.nwb'; 
with NWBHDF5IO(nwb_filename, 'w') as io:
    io.write(nwbfile)

#%%
with NWBHDF5IO(nwb_filename, 'r') as io:
    nwbfile = io.read(); 
    units = nwbfile.units.to_dataframe(); 
    trials = nwbfile.trials.to_dataframe(); 

units['resp_bp'] = ''; 
for i in range(1072):
    psth = units['psth'][i]; 
    bestPos = units['best_pos'][i]; 
    mResp = np.mean(psth[:,100:500],axis=1); 
    best_resp = mResp[np.arange(bestPos,250,5)]; 
    units['resp_bp'][i] = best_resp; 

kkk = np.stack(units[units['session_id']==2].sort_values(by='depth')['resp_bp'].values); 
plt.imshow(np.corrcoef(kkk),origin='lower')

#%% reconstruct stimData into dict
def get_stimDict():

    if os.path.isfile('./stimDict.npy'):
        stimDict = np.load('./stimDict.npy',allow_pickle=True).item(); 
    else:
        stimData = h5py.File('StimData_3scale.mat','r')
        filterData = h5py.File('Filters.mat','r')

        stimDict = {};  
        shapeID_all = []; 
        recon_all = [];        # reconstruction (4 orientation filter output)
        resp4ori_all = [];     # responses of 4 orientation filter

        for i in np.arange(370):
            shapeID_all.append(i); 

            recon = []; 
            for j in np.arange(9,13):
                recon_part = np.reshape(stimData[stimData['StimData']['Recon'][i][0]][j,:],[100,100]).T.flatten(); 
                recon.append(recon_part); 
            recon = np.array(recon); 
            recon_all.append(recon); 

            resp4ori = np.sum(np.abs(recon),1); 
            resp4ori_all.append(resp4ori); 

        stimDict['shapeID'] = shapeID_all; 
        stimDict['recon'] = recon_all; 
        stimDict['resp4ori'] = resp4ori_all; 

        filter_all = []; 
        for j in np.arange(13):
            filter_part = np.reshape(filterData['Filters'][j],[100,100]).T.flatten(); 
            filter_all.append(filter_part); 
        stimDict['filter'] = filter_all; 

        np.save('stimDict.npy',stimDict); 

    return stimDict; 

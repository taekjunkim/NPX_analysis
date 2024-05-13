#%% import modules 
import matplotlib.pyplot as plt;
import scipy.optimize as opt;
import glob
import numpy as np; 
import os; 
#import sys;
#sys.path.append('./helper'); 
#sys.path.append('./makeSDF'); 
import makeSDF;
import parseTJexperiment_NPX as parse_NPX;
import json;
import gzip;


#%% analysis part
def main(app):

    dat_filename = app.dat_file; 
    imec_filename = app.imec_file; 
    task_folder = dat_filename[:(dat_filename.rfind('/')+1)]; 
    bin_filename = glob.glob(task_folder+'*.nidq.bin')[0]; 

    task_index_in_combine = int(app.tasknum_lineEdit.text()); 

    prevTime = 0.3; 
    numStims = 49; 
    experiment = parse_NPX.main(bin_filename, dat_filename, prevTime, numStims, imec_filename, app); 
    experiment['StimDur'] = experiment['stimon']; 

    TimeOfInterest = np.arange(int(prevTime*1000),int(prevTime*1000+experiment['StimDur']+100+1)); 

    StimResp = []; 
    mResp = np.zeros((numStims,experiment['numNeurons']));
    for i in np.arange(len(experiment['stimStructs'])):
        StimResp.append(dict()); 
        StimResp[i]['timeOn'] = experiment['stimStructs'][i]['timeOn']
        StimResp[i]['timeOff'] = experiment['stimStructs'][i]['timeOff']   
        StimResp[i]['pdOn'] = experiment['stimStructs'][i]['pdOn']
        StimResp[i]['pdOff'] = experiment['stimStructs'][i]['pdOff'];       
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

    del experiment['stimStructs']; 
    del experiment['iti_start']; 
    del experiment['iti_end']; 
    #del experiment['iti'];  
    experiment['filename'] = dat_filename; 
    experiment['StimResp'] = StimResp; 
    experiment['mResp'] = mResp; 

    depth = np.array(experiment['chpos_sua'])[:,1]; 
    depth_order = depth.argsort();   # from the tip to the top

    neurons_from_strong = np.sum(mResp,axis=0).argsort()[::-1]; 

    plt.figure(figsize=(8, 4)); 
    ax1 = plt.subplot(1,2,1); 
    ax2 = plt.subplot(1,2,2); 

    nClusters = experiment['numNeurons']; 
    for i in range(nClusters):

        unit_now = neurons_from_strong[i]; 
        if app.all_radiobutton.isChecked() == True:
            unit_id = experiment['neuronid'][unit_now]; 
        elif app.sua_radiobutton.isChecked() == True:
            unit_id = experiment['id_sua'][unit_now]; 
        elif app.mua_radiobutton.isChecked() == True:
            unit_id = experiment['id_mua'][unit_now]; 

        ### Object motion
        ax1.clear()        
        nSteps = [7, 14, 21, 42, 84];         
        for s in np.arange(5):
            condNow = np.arange(s*8, (s+1)*8); 
            ax1.plot(np.arange(0,360,45), mResp[condNow,unit_now],'o-',label=f'nSteps = {nSteps[s]}');
        ax1.spines[['right', 'top']].set_visible(False); 
        ax1.set_xlabel('Direction (deg)');         
        ax1.legend()            
        ax1.set_title('Object motion')            

        ### Surface motion
        ax2.clear()        
        ax2.plot(np.arange(0,360,45), mResp[40:48,unit_now],'ko-'); 
        ax2.spines[['right', 'top']].set_visible(False); 
        ax2.set_xlabel('Direction (deg)');         
        ax2.legend()            
        ax2.set_title('Surface motion')            

        plt.tight_layout(); 
        plt.pause(3); 

        if app.running == 0:
            break; 

    plt.show(block=True); 

    path_to_save = imec_filename[:(imec_filename.rfind('/')+1)] + 'processed/'; 
    if os.path.exists(path_to_save)==0:
        os.mkdir(path_to_save); 

    name_to_save = path_to_save + bin_filename[(bin_filename.rfind('/')+1):-8] + 'json.gz';
    f = gzip.GzipFile(name_to_save,'w');    
    f.write(json.dumps(experiment, cls=NumpyEncoder).encode('utf-8')); 
    f.close(); 
    print('processed file was saved'); 


#%% NumpyEncoder for JSON
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

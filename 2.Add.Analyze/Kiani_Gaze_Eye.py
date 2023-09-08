#%%
import json;
import gzip;
import matplotlib.pyplot as plt;
import numpy as np; 
import pandas as pd; 

#%%
def getSDF(spkTrain,FS,AllorMean=0):
    # Make gaussian kernel window
    sigma = 5;
    t = np.arange(-3*sigma,3*sigma+1);

    y = (1/sigma*np.sqrt(np.pi*2)) * np.exp(-(t**2)/(2*sigma**2));
    window = y[:];
    window = window/np.sum(window);

    # convolution
    sdf = np.zeros(np.shape(spkTrain));
    for i in np.arange(np.shape(spkTrain)[0]):
        convspike = np.convolve(spkTrain[i,:],window);
        pStart = int(np.floor(len(window)/2));
        pEnd = int(np.floor(len(window)/2)+np.shape(spkTrain)[1]);
        convspike = convspike[pStart:pEnd];
        sdf[i,:] = convspike;
    sdf = sdf*FS;

    if AllorMean==0:
        sdf = np.mean(sdf,axis=0);
    
    return sdf;        

#%%
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


#%%
#filename = '/Volumes/TK_exHDD1/NPX/V4_Kiani_Gaze/x230630_combined/processed/x230630_KianiGaze_g1_t1.json.gz'; 
filename = '/Volumes/TK_exHDD1/NPX/V4_Kiani_Gaze/x230703_combined/processed/x230703_KianiObject_g1_t1.json.gz'; 
f = gzip.GzipFile(filename,'r');
Data = json.loads(f.read().decode('utf-8'));
f.close();


#%%
#info_file = '/Volumes/TK_exHDD1/NPX/V4_Kiani_Gaze/x230630_combined/info/imec_datainfo.npy'
info_file = '/Volumes/TK_exHDD1/NPX/V4_Kiani_Gaze/x230703_combined/info/imec_datainfo.npy'
info_data = np.load(info_file, allow_pickle=True).item(); 


#%%
dat_filename = '/Volumes/TK_exHDD1/NPX/V4_Kiani_Gaze/x230703_combined/x230703_KianiObject_g1/x_230703_105454_KianiObject_gaze.dat'; 
fid = open(dat_filename, 'r'); 
markervals_str = []; 
while True:
    tline = fid.readline(); 
    if tline=='':
        break; 
    # remove brackets, single quote. then make a single array
    markervals_str += tline[1:-2].replace("'",'').split(', '); 




#%%
### ch idx0: eyeH
### ch idx1: eyeV
### ch idx2: sync - square wave going to nidq.bin & ap.bin
### ch idx3: pupil
### ch idx4: photodiode
### ch idx5: digital. photodiode (pin0), event (pin1)


#nidq_filename = '/Volumes/TK_exHDD1/NPX/V4_Kiani_Gaze/x230630_combined/x230630_KianiGaze_g1/x230630_KianiGaze_g1_t1.nidq.bin'; 
#meta_filename = '/Volumes/TK_exHDD1/NPX/V4_Kiani_Gaze/x230630_combined/x230630_KianiGaze_g1/x230630_KianiGaze_g1_t1.nidq.meta'; 
nidq_filename = '/Volumes/TK_exHDD1/NPX/V4_Kiani_Gaze/x230703_combined/x230703_KianiObject_g1/x230703_KianiObject_g1_t1.nidq.bin'; 
meta_filename = '/Volumes/TK_exHDD1/NPX/V4_Kiani_Gaze/x230703_combined/x230703_KianiObject_g1/x230703_KianiObject_g1_t1.nidq.meta'; 


meta = get_metaDict(meta_filename); 
nChan = int(meta['nSavedChans'])
nFileSamp = int(int(meta['fileSizeBytes'])/(2*nChan))
nidqData = np.memmap(nidq_filename, dtype='int16', mode='r',
                    shape=(nChan, nFileSamp), offset=0, order='F'); 


niSampRate = 25000
syncCh = int(meta['syncNiChan']); 
nidq_syncONs = np.where(nidqData[syncCh,:niSampRate*20]
                    >np.max(nidqData[syncCh,:niSampRate*20])*0.5)[0];    
for p in range(10):
    if nidq_syncONs[p+10]-nidq_syncONs[p]==10:
        nidq_syncON = nidq_syncONs[p]; 
        break; 

nidq_time = (np.arange(nFileSamp)-nidq_syncON)/niSampRate; 

digitCh = np.shape(nidqData)[0]-1;   # the last channel     
digit_signal = nidqData[digitCh, :]; 
digit_diff = digit_signal[1:] - digit_signal[:-1]; 

### time (ms) with respect to syncON
markerts = (np.where(digit_diff==2)[0] + 1 - nidq_syncON)/niSampRate; 
pdOnTS_raw = (np.where(digit_diff==1)[0] + 1 - nidq_syncON)/niSampRate; 
pdOffTS_raw = (np.where(digit_diff==-1)[0] + 1 - nidq_syncON)/niSampRate; 

### this is for photodiode generating pdOn for every frame
pdOn_dist = pdOnTS_raw[1:] - pdOnTS_raw[:-1];
pdOnTS = np.append(pdOnTS_raw[0],
                    pdOnTS_raw[np.where(pdOn_dist>0.02)[0]+1]);

pdOff_dist = pdOffTS_raw[1:] - pdOffTS_raw[:-1];
pdOffTS = np.append(pdOffTS_raw[np.where(pdOff_dist>0.02)[0]],
                    pdOffTS_raw[-1]); 

#%%
#st = np.load('/Volumes/TK_exHDD1/NPX/V4_Kiani_Gaze/x230630_combined/'+'spike_times.npy'); 
#sc = np.load('/Volumes/TK_exHDD1/NPX/V4_Kiani_Gaze/x230630_combined/'+'spike_clusters.npy'); 
st = np.load('/Volumes/TK_exHDD1/NPX/V4_Kiani_Gaze/x230703_combined/'+'spike_times.npy'); 
sc = np.load('/Volumes/TK_exHDD1/NPX/V4_Kiani_Gaze/x230703_combined/'+'spike_clusters.npy'); 

task_index_in_combine = 1; 

st = np.array(st).astype(int)
ap_syncON = info_data['ap: firstSamp'][task_index_in_combine] + info_data['ap: syncON'][task_index_in_combine]; 

st_unit = (st[np.where(sc==1062)[0]] - ap_syncON)/30000.0; 

#%%
sample_on_str = np.where(np.array(markervals_str) == 'sample_on')[0]; 
figNum = 0; 
for t in np.arange(200,len(pdOnTS)):
    #if ((markervals_str[sample_on_str[t]-4] == 'end_iti') and (int(markervals_str[sample_on_str[t]-1])-200 > 51*2)
    #       and (markervals_str[sample_on_str[t]+4] == 'sample_on') and (markervals_str[sample_on_str[t]+8] == 'sample_on')):
    if ((markervals_str[sample_on_str[t]-4] == 'end_iti') 
           and (markervals_str[sample_on_str[t]+4] == 'sample_on') and (markervals_str[sample_on_str[t]+8] == 'sample_on')):
        
        t1 = int((pdOnTS[t]-10)*1000); 
        t2 = int((pdOnTS[t]+10)*1000); 

        nidq_range = np.where((nidq_time>=t1/1000) & (nidq_time<=t2/1000))[0];  

        plt.figure(figsize=(7,5))
        plt.subplot(3,1,1); 
        plt.plot(nidqData[0,nidq_range],label='eyeH'); 
        plt.plot(nidqData[1,nidq_range],label='eyeV'); 
        plt.ylim([0,40000])
        if (int(markervals_str[sample_on_str[t]-1])-200 > 51*2):
            plt.title(f'stim_id = {int(markervals_str[sample_on_str[t]-1])-200}. t = {t}. Gaze UP' ); 
        else:
            plt.title(f'stim_id = {int(markervals_str[sample_on_str[t]-1])-200}. t = {t}. Gaze DOWN' ); 
        plt.legend()
        plt.xticks(np.arange(0,625000,125000),np.arange(-10,15,5)); 
        plt.gca().spines[['right', 'top']].set_visible(False)

        plt.subplot(3,1,2); 
        stNow = st_unit[np.where((st_unit>=t1/1000) & (st_unit<=t2/1000))[0]]-t1/1000; 
        stNow = np.array(stNow*1000).astype(int); 
        plt.plot(stNow, np.ones(np.shape(stNow)),'k.',label='spikes')

        spk_train = np.zeros((1,(t2-t1))); 
        spk_train[0,stNow] = 1; 
        sdf = getSDF(spk_train,1000); 
        plt.plot(sdf,'g',label='SDF'); 
        plt.xticks(np.arange(0,25000,5000),np.arange(-10,15,5));         
        plt.gca().spines[['right', 'top']].set_visible(False)                

        plt.subplot(3,1,3); 
        plt.plot(nidqData[4,nidq_range]); 
        plt.xticks(np.arange(0,625000,125000),np.arange(-1,1.5,0.5));     
        plt.xlabel('Time (sec)')    
        plt.gca().spines[['right', 'top']].set_visible(False)                
        
        plt.tight_layout(); 


        figNum += 1; 
    if figNum > 35:
        break; 





# %%
sample_on_str = np.where(np.array(markervals_str) == 'sample_on')[0]; 
kernel_size = 10
kernel = np.ones(kernel_size) / kernel_size

eye_PD_t_fix = np.empty((0,4)); 
for t in np.arange(len(pdOnTS)):
    if ((markervals_str[sample_on_str[t]-4] == 'end_iti') 
           and (markervals_str[sample_on_str[t]+4] == 'sample_on') and (markervals_str[sample_on_str[t]+8] == 'sample_on')):
        
        nidq_rangeA = np.where((nidq_time>=pdOnTS[t]-2) & (nidq_time<=pdOnTS[t]))[0];  
        eyeH_abs = np.abs(np.diff(np.convolve(nidqData[0,nidq_rangeA],kernel))); 
        eyeV_abs = np.abs(np.diff(np.convolve(nidqData[1,nidq_rangeA],kernel))); 
        eyeH_abs[:10] = 0; eyeH_abs[-10:] = 0; 
        eyeV_abs[:10] = 0; eyeV_abs[-10:] = 0; 
        eye_ON = nidq_time[nidq_rangeA[np.where((eyeH_abs+eyeV_abs)>15)[0][-1]]]; 

        #t1 = int((eye_ON-1)*1000); 
        #t2 = int((eye_ON+2)*1000); 
        #nidq_range = np.where((nidq_time>=t1/1000) & (nidq_time<=t2/1000))[0];  

        if (int(markervals_str[sample_on_str[t]-1])-200 < 51):
            eye_PD_t_fix = np.vstack((eye_PD_t_fix,[eye_ON, pdOnTS[t],t,1])); 
        elif (int(markervals_str[sample_on_str[t]-1])-200 < 51*2):
            eye_PD_t_fix = np.vstack((eye_PD_t_fix,[eye_ON, pdOnTS[t],t,2])); 
        elif (int(markervals_str[sample_on_str[t]-1])-200 < 51*3):
            eye_PD_t_fix = np.vstack((eye_PD_t_fix,[eye_ON, pdOnTS[t],t,3])); 
        else:
            eye_PD_t_fix = np.vstack((eye_PD_t_fix,[eye_ON, pdOnTS[t],t,4])); 
        

df = pd.DataFrame(eye_PD_t_fix); 
df.columns = ['eyeON', 'pdON', 'tNum', 'fixPosition']; 
df['Eye_to_PD'] = df['pdON'] - df['eyeON']; 
df = df.sort_values(by='Eye_to_PD').reset_index(drop=True); 

#%%
#st = np.load('/Volumes/TK_exHDD1/NPX/V4_Kiani_Gaze/x230630_combined/'+'spike_times.npy'); 
#sc = np.load('/Volumes/TK_exHDD1/NPX/V4_Kiani_Gaze/x230630_combined/'+'spike_clusters.npy'); 
st = np.load('/Volumes/TK_exHDD1/NPX/V4_Kiani_Gaze/x230703_combined/'+'spike_times.npy'); 
sc = np.load('/Volumes/TK_exHDD1/NPX/V4_Kiani_Gaze/x230703_combined/'+'spike_clusters.npy'); 

task_index_in_combine = 1; 

st = np.array(st).astype(int)
ap_syncON = info_data['ap: firstSamp'][task_index_in_combine] + info_data['ap: syncON'][task_index_in_combine]; 

unitID = 1062;  # 991, 1062
st_unit = (st[np.where(sc==unitID)[0]] - ap_syncON)/30000.0; 


# aligned at FixON
pos1_spkA = np.empty((0,3*1000)); 
pos2_spkA = np.empty((0,3*1000)); 
pos3_spkA = np.empty((0,3*1000)); 
pos4_spkA = np.empty((0,3*1000)); 
# aligned at StimON
pos1_spkB = np.empty((0,3*1000)); 
pos2_spkB = np.empty((0,3*1000)); 
pos3_spkB = np.empty((0,3*1000)); 
pos4_spkB = np.empty((0,3*1000)); 


plt.figure(figsize=(15,5)); 
plt.gcf().suptitle(f"unitID = {unitID}. SUA", fontsize=16)

ax1 = plt.subplot(2,4,1); 
ax2 = plt.subplot(2,4,2); 
ax3 = plt.subplot(2,4,3); 
ax4 = plt.subplot(2,4,4); 
ax5 = plt.subplot(2,4,5); 
ax6 = plt.subplot(2,4,6); 
ax7 = plt.subplot(2,4,7); 
ax8 = plt.subplot(2,4,8); 

for i in range(len(df)):

    ### eye align
    t1 = df.loc[i,'eyeON']-1; 
    t2 = df.loc[i,'eyeON']+2; 
    
    stNow = (st_unit[np.where((st_unit>=t1) & (st_unit<=t2))[0]]-t1)*1000; 
    stNow = np.array(stNow).astype(int).flatten(); 

    spkNow = np.zeros((1,3000)); 
    spkNow[0,stNow] = 1; 
    if df.loc[i,'fixPosition']==1:
        pos1_spkA = np.vstack((pos1_spkA,spkNow)); 
        ax5.plot(stNow, np.ones(np.shape(stNow))*np.shape(pos1_spkA)[0],'k.',ms=2); 
        ax5.plot(1000,np.shape(pos1_spkA)[0],'b.',ms=2); 
        ax5.plot(1000+df.loc[i,'Eye_to_PD']*1000,np.shape(pos1_spkA)[0],'r.',ms=2);         
        ax5.plot(1600+df.loc[i,'Eye_to_PD']*1000,np.shape(pos1_spkA)[0],'r.',ms=2);         
        ax5.plot(2200+df.loc[i,'Eye_to_PD']*1000,np.shape(pos1_spkA)[0],'r.',ms=2);                 
    elif df.loc[i,'fixPosition']==2:
        pos2_spkA = np.vstack((pos2_spkA,spkNow)); 
        ax6.plot(stNow, np.ones(np.shape(stNow))*np.shape(pos2_spkA)[0],'k.',ms=2);     
        ax6.plot(1000,np.shape(pos2_spkA)[0],'b.',ms=2);         
        ax6.plot(1000+df.loc[i,'Eye_to_PD']*1000,np.shape(pos2_spkA)[0],'r.',ms=2);         
        ax6.plot(1600+df.loc[i,'Eye_to_PD']*1000,np.shape(pos2_spkA)[0],'r.',ms=2);         
        ax6.plot(2200+df.loc[i,'Eye_to_PD']*1000,np.shape(pos2_spkA)[0],'r.',ms=2);                 
    elif df.loc[i,'fixPosition']==3:
        pos3_spkA = np.vstack((pos3_spkA,spkNow)); 
        ax2.plot(stNow, np.ones(np.shape(stNow))*np.shape(pos3_spkA)[0],'k.',ms=2);     
        ax2.plot(1000,np.shape(pos3_spkA)[0],'b.',ms=2);                 
        ax2.plot(1000+df.loc[i,'Eye_to_PD']*1000,np.shape(pos3_spkA)[0],'r.',ms=2);         
        ax2.plot(1600+df.loc[i,'Eye_to_PD']*1000,np.shape(pos3_spkA)[0],'r.',ms=2);         
        ax2.plot(2200+df.loc[i,'Eye_to_PD']*1000,np.shape(pos3_spkA)[0],'r.',ms=2);                 
    elif df.loc[i,'fixPosition']==4:
        pos4_spkA = np.vstack((pos4_spkA,spkNow)); 
        ax1.plot(stNow, np.ones(np.shape(stNow))*np.shape(pos4_spkA)[0],'k.',ms=2);    
        ax1.plot(1000,np.shape(pos4_spkA)[0],'b.',ms=2);                      
        ax1.plot(1000+df.loc[i,'Eye_to_PD']*1000,np.shape(pos4_spkA)[0],'r.',ms=2);         
        ax1.plot(1600+df.loc[i,'Eye_to_PD']*1000,np.shape(pos4_spkA)[0],'r.',ms=2);         
        ax1.plot(2200+df.loc[i,'Eye_to_PD']*1000,np.shape(pos4_spkA)[0],'r.',ms=2);                 


    ### stim align
    t1 = df.loc[i,'pdON']-1; 
    t2 = df.loc[i,'pdON']+2; 
    
    stNow = (st_unit[np.where((st_unit>=t1) & (st_unit<=t2))[0]]-t1)*1000; 
    stNow = np.array(stNow).astype(int).flatten(); 

    spkNow = np.zeros((1,3000)); 
    spkNow[0,stNow] = 1; 
    if df.loc[i,'fixPosition']==1:
        pos1_spkB = np.vstack((pos1_spkB,spkNow)); 
        ax7.plot(stNow, np.ones(np.shape(stNow))*np.shape(pos1_spkB)[0],'k.',ms=2); 
        ax7.plot(1000,np.shape(pos1_spkB)[0],'r.',ms=2); 
        ax7.plot(1600,np.shape(pos1_spkB)[0],'r.',ms=2); 
        ax7.plot(2200,np.shape(pos1_spkB)[0],'r.',ms=2);                 
        ax7.plot(1000-df.loc[i,'Eye_to_PD']*1000,np.shape(pos1_spkB)[0],'b.',ms=2);     
    elif df.loc[i,'fixPosition']==2:
        pos2_spkB = np.vstack((pos2_spkB,spkNow)); 
        ax8.plot(stNow, np.ones(np.shape(stNow))*np.shape(pos2_spkB)[0],'k.',ms=2); 
        ax8.plot(1000,np.shape(pos2_spkB)[0],'r.',ms=2); 
        ax8.plot(1600,np.shape(pos2_spkB)[0],'r.',ms=2); 
        ax8.plot(2200,np.shape(pos2_spkB)[0],'r.',ms=2);                 
        ax8.plot(1000-df.loc[i,'Eye_to_PD']*1000,np.shape(pos2_spkB)[0],'b.',ms=2);     
    elif df.loc[i,'fixPosition']==3:
        pos3_spkB = np.vstack((pos3_spkB,spkNow)); 
        ax4.plot(stNow, np.ones(np.shape(stNow))*np.shape(pos3_spkB)[0],'k.',ms=2); 
        ax4.plot(1000,np.shape(pos3_spkB)[0],'r.',ms=2); 
        ax4.plot(1600,np.shape(pos3_spkB)[0],'r.',ms=2); 
        ax4.plot(2200,np.shape(pos3_spkB)[0],'r.',ms=2);                 
        ax4.plot(1000-df.loc[i,'Eye_to_PD']*1000,np.shape(pos3_spkB)[0],'b.',ms=2);     
    elif df.loc[i,'fixPosition']==4:
        pos4_spkB = np.vstack((pos4_spkB,spkNow)); 
        ax3.plot(stNow, np.ones(np.shape(stNow))*np.shape(pos4_spkB)[0],'k.',ms=2); 
        ax3.plot(1000,np.shape(pos4_spkB)[0],'r.',ms=2); 
        ax3.plot(1600,np.shape(pos4_spkB)[0],'r.',ms=2); 
        ax3.plot(2200,np.shape(pos4_spkB)[0],'r.',ms=2);                 
        ax3.plot(1000-df.loc[i,'Eye_to_PD']*1000,np.shape(pos4_spkB)[0],'b.',ms=2);     


ax1.plot(getSDF(pos4_spkA,1000)*5,'g',lw=2); 
ax1.set_title('Gaze position: (0, 5)'); 
ax2.plot(getSDF(pos3_spkA,1000)*5,'g',lw=2); 
ax2.set_title('Gaze position: (5, 5)'); 
ax5.plot(getSDF(pos1_spkA,1000)*5,'g',lw=2); 
ax5.set_title('Gaze position: (0, 0)'); 
ax6.plot(getSDF(pos2_spkA,1000)*5,'g',lw=2); 
ax6.set_title('Gaze position: (5, 0)'); 

ax3.plot(getSDF(pos4_spkB,1000)*5,'g',lw=2); 
ax3.set_title('Gaze position: (0, 5)'); 
ax4.plot(getSDF(pos3_spkB,1000)*5,'g',lw=2); 
ax4.set_title('Gaze position: (5, 5)'); 
ax7.plot(getSDF(pos1_spkB,1000)*5,'g',lw=2); 
ax7.set_title('Gaze position: (0, 0)'); 
ax8.plot(getSDF(pos2_spkB,1000)*5,'g',lw=2); 
ax8.set_title('Gaze position: (5, 0)'); 


for i in range(1,9):
    eval(f'ax{i}.set_xticks(np.arange(0,4000,1000),np.arange(-1000,3000,1000))'); 
    eval(f'ax{i}.set_yticks(np.arange(0,150,50),np.arange(0,30,10))');     
    eval(f'ax{i}.set_xlim([0,3000])'); 
    eval(f"ax{i}.spines[['right', 'top']].set_visible(False)"); 

plt.tight_layout(); 
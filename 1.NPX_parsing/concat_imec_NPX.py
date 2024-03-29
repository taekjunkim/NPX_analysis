#%%
import sys
sys.path.append('./helper'); 

from PyQt5.QtCore import Qt; 
from PyQt5 import QtWidgets; 
from PyQt5.QtWidgets import QApplication, QMainWindow; 
import numpy as np; 
import glob; 
import os; 


#%%
# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NPX concatenation"); 
        main_frame = QtWidgets.QWidget()
        self.setCentralWidget(main_frame)
        self.initUI(); 

    def initUI(self):
        layout = QtWidgets.QVBoxLayout()

        layout.addWidget(self.task_selection_group())
        #layout.addStretch(1)
        layout.addWidget(self.destination_folder_group())

        self.run_button = QtWidgets.QPushButton(); 
        self.run_button.setText("Run!!"); 
        self.run_button.clicked.connect(self.run_button_clicked); 
        layout.addWidget(self.run_button)

        self.centralWidget().setLayout(layout); 

    def task_selection_group(self):
        groupBox = QtWidgets.QGroupBox("List of ap.bin files"); 

        self.append_button = QtWidgets.QPushButton(); 
        self.append_button.setText("Append"); 
        self.append_button.clicked.connect(self.append_button_clicked); 

        self.remove_button = QtWidgets.QPushButton(); 
        self.remove_button.setText("Remove"); 
        self.remove_button.clicked.connect(self.remove_button_clicked); 

        self.task_list = QtWidgets.QListWidget() 

        self.moveup_button = QtWidgets.QToolButton(); 
        self.moveup_button.setArrowType(Qt.UpArrow); 
        self.moveup_button.clicked.connect(self.moveup_button_clicked); 

        self.movedown_button = QtWidgets.QToolButton(); 
        self.movedown_button.setArrowType(Qt.DownArrow);
        self.movedown_button.clicked.connect(self.movedown_button_clicked); 

        grid = QtWidgets.QGridLayout()
        grid.addWidget(self.append_button,0,0); 
        grid.addWidget(self.remove_button,0,1); 
        grid.addWidget(self.task_list,1,0,3,2);         
        grid.addWidget(self.moveup_button,1,2); 
        grid.addWidget(self.movedown_button,3,2); 
        groupBox.setLayout(grid)

        return groupBox; 

    def destination_folder_group(self):
        groupBox = QtWidgets.QGroupBox("Destination folder name: check g#");
        self.tasknum_lineEdit = QtWidgets.QLineEdit(); 
        vbox = QtWidgets.QVBoxLayout(); 
        vbox.addWidget(self.tasknum_lineEdit); 
        groupBox.setLayout(vbox); 

        return groupBox; 

    def append_button_clicked(self):
        imec_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Select an imec ap.bin file')
        self.task_list.addItem(imec_path[0]); 

        if len(self.tasknum_lineEdit.text())==0:
            slashes = [idx for idx, character in enumerate(imec_path[0]) if character=='/']; 
            new_dirname = imec_path[0][:slashes[-3]+8] + '_combined_g#'
            self.tasknum_lineEdit.setText(new_dirname); 
        else:    
            pass; 

    def remove_button_clicked(self):        
        listItems = self.task_list.selectedItems()
        if not listItems: return        
        for item in listItems:
           self.task_list.takeItem(self.task_list.row(item));  
        
    def moveup_button_clicked(self):        
        currentRow = self.task_list.currentRow();
        currentItem = self.task_list.takeItem(currentRow);
        self.task_list.insertItem(currentRow - 1, currentItem);

    def movedown_button_clicked(self):                
        currentRow = self.task_list.currentRow();
        currentItem = self.task_list.takeItem(currentRow);
        self.task_list.insertItem(currentRow + 1, currentItem);

    def run_button_clicked(self):                
        new_dirname = self.tasknum_lineEdit.text(); 
        imecs = self.task_list.item(0).text()[-12:-7]; 
        new_filename = new_dirname[(new_dirname.rfind('/')+1):]; 
        new_ap_name = new_dirname + '/' + new_filename + '_' + imecs + '.ap.bin'; 
        new_lf_name = new_dirname + '/' + new_filename + '_' + imecs + '.lf.bin'; 
        datainfo_name = new_dirname + '/info/' + 'imec_datainfo.npy';  

        # make directory for the concatenated file
        if os.path.exists(new_dirname)==0:
            os.mkdir(new_dirname); 
            os.mkdir(new_dirname+'/info'); 

        # get_imec_datainfo
        imec_datainfo = get_imec_datainfo(self); 
        np.save(datainfo_name, imec_datainfo); 


        #############################
        ### concatenate bin files ###
        #############################

        for i in range(len(imec_datainfo['ap: fname'])):
        
            # get nidq_data
            nidq_name = imec_datainfo['nidq: fname'][i]; 
            nidq_nFileSamp = imec_datainfo['nidq: nFileSamp'][i]; 
            nidq_nChan = imec_datainfo['nidq: nChan'][i]; 
            nidq_syncCH = imec_datainfo['nidq: syncCH'][i]; 
            nidq_SampRate = imec_datainfo['nidq: SampRate'][i]; 
            nidq_data = np.memmap(nidq_name, dtype='int16', 
                                shape=(nidq_nFileSamp, nidq_nChan), offset=0, order='C'); 

            # get ap_data
            ap_name = imec_datainfo['ap: fname'][i]; 
            ap_nFileSamp = imec_datainfo['ap: nFileSamp'][i]; 
            ap_nChan = imec_datainfo['ap: nChan'][i]; 
            ap_imSampRate = imec_datainfo['ap: imSampRate'][i]; 
            ap_data = np.memmap(ap_name, dtype='int16', 
                                shape=(ap_nFileSamp, ap_nChan), offset=0, order='C'); 

            # get lf_data                            
            lf_name = imec_datainfo['lf: fname'][i]; 
            lf_nFileSamp = imec_datainfo['lf: nFileSamp'][i]; 
            lf_nChan = imec_datainfo['lf: nChan'][i]; 
            lf_imSampRate = imec_datainfo['lf: imSampRate'][i];         
            lf_data = np.memmap(lf_name, dtype='int16', 
                                shape=(lf_nFileSamp, lf_nChan), offset=0, order='C');                                           

            # get_syncON in the first and last 20 seconds
            nidq_syncONs_A = np.where(
                nidq_data[:nidq_SampRate*20, nidq_syncCH]
                > np.max(nidq_data[:nidq_SampRate*20, nidq_syncCH])*0.5
            )[0];    
            nidq_syncONs_B = np.where(
                nidq_data[-nidq_SampRate*20:, nidq_syncCH]
                > np.max(nidq_data[-nidq_SampRate*20:, nidq_syncCH])*0.5
            )[0] + nidq_nFileSamp - nidq_SampRate*20;    
            for p in range(10):
                if nidq_syncONs_A[p+10]-nidq_syncONs_A[p]==10:
                    nidq_syncON_A = nidq_syncONs_A[p]; 
                    break; 
            for p in range(10):
                if nidq_syncONs_B[-p-1]-nidq_syncONs_B[-p-11]==10:
                    nidq_syncON_B = nidq_syncONs_B[-p-1]; 
                    break; 

            syncCH = 384; 
            ap_syncONs_A = np.where(ap_data[:ap_imSampRate*20, syncCH]
                                    >np.max(ap_data[:ap_imSampRate*20, syncCH])*0.5)[0];    
            ap_syncONs_B = np.where(
                ap_data[-ap_imSampRate*20:, syncCH]
                > np.max(ap_data[-ap_imSampRate*20:, syncCH])*0.5
            )[0] + ap_nFileSamp - ap_imSampRate*20;    
            for p in range(10):
                if ap_syncONs_A[p+10]-ap_syncONs_A[p]==10:
                    ap_syncON_A = ap_syncONs_A[p]; 
                    break; 
            for p in range(10):
                if ap_syncONs_B[-p-1]-ap_syncONs_B[-p-11]==10:
                    ap_syncON_B = ap_syncONs_B[-p-1]; 
                    break; 
            
            lf_syncONs_A = np.where(lf_data[:lf_imSampRate*20, syncCH]
                                    >np.max(lf_data[:lf_imSampRate*20, syncCH])*0.5)[0];    
            lf_syncONs_B = np.where(
                lf_data[-lf_imSampRate*20:, syncCH]
                > np.max(lf_data[-lf_imSampRate*20:, syncCH])*0.5
            )[0] + lf_nFileSamp - lf_imSampRate*20;    
            for p in range(10):
                if lf_syncONs_A[p+10]-lf_syncONs_A[p]==10:
                    lf_syncON_A = lf_syncONs_A[p]; 
                    break; 
            for p in range(10):
                if lf_syncONs_B[-p-1]-lf_syncONs_B[-p-11]==10:
                    lf_syncON_B = lf_syncONs_B[-p-1]; 
                    break; 

            imec_datainfo['nidq: syncON'].append(nidq_syncON_A); 
            imec_datainfo['ap: syncON'].append(ap_syncON_A); 
            imec_datainfo['lf: syncON'].append(lf_syncON_A);                                      

            imec_datainfo['nidq: syncOFF'].append(nidq_syncON_B); 
            imec_datainfo['ap: syncOFF'].append(ap_syncON_B); 
            imec_datainfo['lf: syncOFF'].append(lf_syncON_B);                                      


            # write ap_data, lf_data
            chunk = 0; 
            ap_chunk_size = int(np.ceil(ap_nFileSamp/1000)); 
            lf_chunk_size = int(np.ceil(lf_nFileSamp/1000)); 

            f_ap = open(new_ap_name, 'a'); 
            f_lf = open(new_lf_name, 'a');

            while chunk<1000:

                # ap: 1/1000 chunk of data
                ap_chunk_start = chunk*ap_chunk_size; 
                ap_chunk_end = np.min([(chunk+1)*ap_chunk_size, ap_nFileSamp]); 
                ap_data[ap_chunk_start:ap_chunk_end, :].tofile(f_ap); 

                # lf: 1/1000 chunk of data
                lf_chunk_start = chunk*lf_chunk_size; 
                lf_chunk_end = np.min([(chunk+1)*lf_chunk_size, lf_nFileSamp]); 
                lf_data[lf_chunk_start:lf_chunk_end, :].tofile(f_lf); 

                if chunk==0:
                    print(f'File# {i}: concatenate was initiated'); 
                elif chunk%100==0:
                    print(f'File# {i}: {chunk/10}% was concatenated'); 
                    f_ap.close(); 
                    f_lf.close(); 
                    f_ap = open(new_ap_name, 'a'); 
                    f_lf = open(new_lf_name, 'a');
                chunk += 1; 

            f_ap.close(); 
            f_lf.close(); 


        # remove memmap
        del nidq_data, ap_data, lf_data 

        np.save(datainfo_name, imec_datainfo); 
        print('imec_datainfo was saved'); 
        pass; 


def get_imec_datainfo(app):

    ### internal structure will be defined in the for loop
    imec_datainfo = dict(); 
    imec_datainfo['nidq: fname'] = []; 
    imec_datainfo['nidq: nFileSamp'] = []; 
    imec_datainfo['nidq: syncCH'] = [];     
    imec_datainfo['nidq: syncON'] = [];         # syncON in a single file
    imec_datainfo['nidq: syncOFF'] = [];         # syncON in a single file    
    imec_datainfo['nidq: nChan'] = []; 
    imec_datainfo['nidq: SampRate'] = [];  

    imec_datainfo['ap: fname'] = []; 
    imec_datainfo['ap: nFileSamp'] = []; 
    imec_datainfo['ap: firstSamp'] = [];     
    imec_datainfo['ap: syncON'] = [];         # syncON in a single file
    imec_datainfo['ap: syncOFF'] = [];         # syncON in a single file    
    imec_datainfo['ap: syncON_concat'] = [];  # syncON in a concatenated file             
    imec_datainfo['ap: nChan'] = []; 
    imec_datainfo['ap: imSampRate'] = [];  

    imec_datainfo['lf: fname'] = []; 
    imec_datainfo['lf: nFileSamp'] = []; 
    imec_datainfo['lf: firstSamp'] = [];       
    imec_datainfo['lf: syncON'] = [];         # syncON in a single file     
    imec_datainfo['lf: syncOFF'] = [];         # syncON in a single file         
    imec_datainfo['lf: syncON_concat'] = [];  # syncON in a concatenated file              
    imec_datainfo['lf: nChan'] = []; 
    imec_datainfo['lf: imSampRate'] = [];  

    ap_bins = []; 
    lf_bins = []; 
    for i in range(app.task_list.count()):
        ap_bins.append(app.task_list.item(i).text()); 
        lf_bins.append(app.task_list.item(i).text()[:-6]+'lf.bin'); 

    for j in range(len(ap_bins)):

        ### read meta for nadq, ap and lf
        idx_slash = []; 
        for c in np.arange(len(ap_bins[j])):
            if ap_bins[j][c]=='/':
                idx_slash.append(c); 

        nidq_bin = glob.glob(ap_bins[j][:idx_slash[-2]]+'/*nidq.bin')[0]
        nidq_metaname = nidq_bin[:-3]+'meta'; 
        nidq_meta = get_metaDict(nidq_metaname); 
        nidq_syncCH = int(nidq_meta['syncNiChan'])
        nidq_nChan = int(nidq_meta['nSavedChans']); 
        nidq_nFileSamp = int(int(nidq_meta['fileSizeBytes'])/(2*nidq_nChan)); 
        nidq_SampRate = int(nidq_meta['niSampRate']); 

        ap_metaname = ap_bins[j][:-3]+'meta'; 
        ap_meta = get_metaDict(ap_metaname); 
        ap_nChan = int(ap_meta['nSavedChans']); 
        ap_nFileSamp = int(int(ap_meta['fileSizeBytes'])/(2*ap_nChan)); 
        ap_imSampRate = int(ap_meta['imSampRate']); 

        lf_metaname = lf_bins[j][:-3]+'meta';         
        lf_meta = get_metaDict(lf_metaname); 
        lf_nChan = int(lf_meta['nSavedChans']); 
        lf_nFileSamp = int(int(lf_meta['fileSizeBytes'])/(2*lf_nChan)); 
        lf_imSampRate = int(lf_meta['imSampRate']);        

        ### define imec_datainfo
        imec_datainfo['nidq: fname'].append(nidq_bin); 
        imec_datainfo['nidq: nFileSamp'].append(nidq_nFileSamp); 
        imec_datainfo['nidq: nChan'].append(nidq_nChan); 
        imec_datainfo['nidq: SampRate'].append(nidq_SampRate);  
        imec_datainfo['nidq: syncCH'].append(nidq_syncCH);    
        
        imec_datainfo['ap: fname'].append(ap_bins[j]); 
        imec_datainfo['ap: nFileSamp'].append(ap_nFileSamp); 
        if len(imec_datainfo['ap: firstSamp'])==0:
            imec_datainfo['ap: firstSamp'].append(0); 
        else:
            firstSamp = np.sum(imec_datainfo['ap: nFileSamp'][:-1]); 
            imec_datainfo['ap: firstSamp'].append(firstSamp); 
        imec_datainfo['ap: nChan'].append(ap_nChan); 
        imec_datainfo['ap: imSampRate'].append(ap_imSampRate);             

        imec_datainfo['lf: fname'].append(lf_bins[j]); 
        imec_datainfo['lf: nFileSamp'].append(lf_nFileSamp); 
        if len(imec_datainfo['lf: firstSamp'])==0:
            imec_datainfo['lf: firstSamp'].append(0); 
        else:
            firstSamp = np.sum(imec_datainfo['lf: nFileSamp'][:-1]); 
            imec_datainfo['lf: firstSamp'].append(firstSamp); 
        imec_datainfo['lf: nChan'].append(lf_nChan);             
        imec_datainfo['lf: imSampRate'].append(lf_imSampRate);             

    return imec_datainfo;         

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



def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    # Start the event loop.
    app.exec()

#%%
if __name__ == '__main__':
    main(); 


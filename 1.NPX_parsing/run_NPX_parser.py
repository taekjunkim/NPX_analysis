#%%
import sys;
sys.path.append('./helper'); 

from PyQt5 import QtWidgets; 
from PyQt5.QtWidgets import QApplication, QMainWindow; 


#%%
# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NPX Parser"); 
        main_frame = QtWidgets.QWidget()
        self.setCentralWidget(main_frame)
        self.initUI(); 
        self.running = 1; 

    def initUI(self):
        layout = QtWidgets.QVBoxLayout()

        layout.addWidget(self.task_selection_group())
        layout.addStretch(1)
        layout.addWidget(self.unit_selection_group())
        layout.addStretch(1)
        layout.addWidget(self.imec_folder_group())
        layout.addStretch(1)
        layout.addWidget(self.dat_file_group())
        layout.addStretch(1)
        layout.addWidget(self.execute_group())

        self.centralWidget().setLayout(layout); 

    def execute_group(self):
        groupBox = QtWidgets.QGroupBox("Execution"); 

        self.run_button = QtWidgets.QPushButton(); 
        self.run_button.setText("Run!!"); 
        self.run_button.clicked.connect(self.run_button_clicked); 

        self.stop_button = QtWidgets.QPushButton(); 
        self.stop_button.setText("Stop!!"); 
        self.stop_button.clicked.connect(self.stop_button_clicked); 

        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.run_button)                
        #hbox.addStretch(1)
        hbox.addWidget(self.stop_button)                
        groupBox.setLayout(hbox)

        return groupBox; 

    def task_selection_group(self):
        groupBox = QtWidgets.QGroupBox("Task name"); 

        self.combobox = QtWidgets.QComboBox(); 
        self.combobox.addItems(['None']); 
        self.combobox.addItems(['ExtractWaveform']);         
        self.combobox.addItems(['KS_DriftMap']);                 
        self.combobox.addItems(['CSD']); 
        self.combobox.addItems(['DriftingGrating']); 
        self.combobox.addItems(['RndDotRFmap']);         
        self.combobox.addItems(['ClutterStim']);          
        self.combobox.addItems(['SaliencyPS']);          
        self.combobox.addItems(['ShapeTexture']);                  
        self.combobox.addItems(['PositionInvariance']);  
        self.combobox.addItems(['Texture3Ver']);                               
        self.combobox.addItems(['StimDuration']);                                         
        self.combobox.addItems(['SurroundMap']);                                                 
        self.combobox.addItems(['Kiani_gaze']);                        
        self.combobox.addItems(['TextureFlow']);                                                                 

        self.tasknum_label = QtWidgets.QLabel(); 
        self.tasknum_label.setText('task idx in combined file')

        self.tasknum_lineEdit = QtWidgets.QLineEdit(); 
        self.tasknum_lineEdit.setFixedWidth(30); 
        self.tasknum_lineEdit.setText('0'); 

        self.sorted_checkbox = QtWidgets.QCheckBox("manually sorted?"); 

        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.combobox)
        hbox.addStretch(1)
        hbox.addWidget(self.tasknum_label)
        hbox.addWidget(self.tasknum_lineEdit)                
        hbox.addStretch(1)
        hbox.addWidget(self.sorted_checkbox)                
        groupBox.setLayout(hbox)

        return groupBox; 

    def unit_selection_group(self):
        groupBox = QtWidgets.QGroupBox("SUA | MUA | All"); 

        self.sua_radiobutton = QtWidgets.QRadioButton("select SUA"); 
        self.mua_radiobutton = QtWidgets.QRadioButton("select MUA");         
        self.all_radiobutton = QtWidgets.QRadioButton("select All");                 
        self.all_radiobutton.setChecked(True); 

        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.sua_radiobutton)                
        #hbox.addStretch(1)
        hbox.addWidget(self.mua_radiobutton)                
        hbox.addWidget(self.all_radiobutton)                        
        groupBox.setLayout(hbox)

        return groupBox; 

    def imec_folder_group(self):
        groupBox = QtWidgets.QGroupBox("imec file"); 

        self.imec_button = QtWidgets.QPushButton(); 
        self.imec_button.setText("Select an imec ap.bin file  (lf.bin for CSD)"); 
        self.imec_button.clicked.connect(self.imec_button_clicked); 

        self.imec_lineEdit = QtWidgets.QLineEdit(); 

        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.imec_lineEdit)                
        hbox.addWidget(self.imec_button)
        #hbox.addStretch(1)
        groupBox.setLayout(hbox)

        return groupBox; 

    def dat_file_group(self):
        groupBox = QtWidgets.QGroupBox("dat file"); 

        self.dat_button = QtWidgets.QPushButton(); 
        self.dat_button.setText("Select a dat file"); 
        self.dat_button.clicked.connect(self.dat_button_clicked); 
        self.dat_lineEdit = QtWidgets.QLineEdit(); 

        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.dat_lineEdit)        
        hbox.addWidget(self.dat_button)
        #hbox.addStretch(1)
        groupBox.setLayout(hbox)

        return groupBox; 

    def imec_button_clicked(self):
        imec_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Select an imec ap.bin file (lf.bin for CSD)')
        self.imec_file = imec_path[0]; 
        self.imec_lineEdit.setText(imec_path[0]); 

    def dat_button_clicked(self):
        dat_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Select a dat file')
        self.dat_file = dat_path[0]; 
        self.dat_lineEdit.setText(dat_path[0]); 

    def run_button_clicked(self):
        self.running = 1; 
        if self.combobox.currentText() == 'CSD':
            from tasks import CSD_NPX_Analysis as CSD_NPX; 
            CSD_NPX.main(self); 
        if self.combobox.currentText() == 'ExtractWaveform':
            from tasks import ExtractWaveform_NPX_Analysis as EW_NPX; 
            EW_NPX.main(self); 
        if self.combobox.currentText() == 'KS_DriftMap':
            from tasks import KS_DriftMap_NPX_Analysis as KS_Drift_NPX; 
            KS_Drift_NPX.main(self);         
        elif self.combobox.currentText() == 'DriftingGrating':
            from tasks import DriftingGrating_NPX_Analysis as DG_NPX; 
            DG_NPX.main(self); 
        elif self.combobox.currentText() == 'RndDotRFmap':      
            from tasks import RndDotRFmap_NPX_Analysis as RndDot_NPX; 
            RndDot_NPX.main(self); 
        elif self.combobox.currentText() == 'ClutterStim':      
            from tasks import ClutterStim_NPX_Analysis as CS_NPX; 
            CS_NPX.main(self); 
        elif self.combobox.currentText() == 'ShapeTexture':      
            from tasks import ShapeTexture_NPX_Analysis as ST_NPX; 
            ST_NPX.main(self);         
        elif self.combobox.currentText() == 'SaliencyPS':      
            from tasks import SaliencyPSstat_NPX_Analysis as SP_NPX; 
            SP_NPX.main(self); 
        elif self.combobox.currentText() == 'PositionInvariance':      
            from tasks import PositionInvariance_NPX_Analysis as PI_NPX; 
            PI_NPX.main(self); 
        elif self.combobox.currentText() == 'Texture3Ver':      
            from tasks import Texture3Ver_NPX_Analysis as T3V_NPX; 
            T3V_NPX.main(self); 
        elif self.combobox.currentText() == 'StimDuration':      
            print('Not ready yet')
        elif self.combobox.currentText() == 'SurroundMap':                                          
            print('Not ready yet')
        elif self.combobox.currentText() == 'Kiani_gaze':                                          
            from tasks import KianiGaze_NPX_Analysis as kGaze_NPX; 
            kGaze_NPX.main(self); 
        elif self.combobox.currentText() == 'TextureFlow':                                          
            from tasks import TextureFlow_NPX_Analysis as TexFlow_NPX; 
            TexFlow_NPX.main(self);         
        else:
            print('Task was not selected'); 

    def stop_button_clicked(self):
        self.running = 0; 


#%%
def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    # Start the event loop.
    app.exec()

#%%
if __name__ == '__main__':
    main(); 

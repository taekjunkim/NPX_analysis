### How to use
1. In the terminal, move to this folder
   - For example, "/Users/taekjunkim/Documents/UW_PasupathyLab/NPX_analysis/1.NPX_parsing"
2. Type 
'''
python run_NPX_parser.py
'''


### Pre-processing procedure ###

1. Sort multiple bin files in chronological order
   - nidq.bin
   - ap.bin

2. Make merged ap.bin
   - Get nFileSamp, syncON from each ap.bin

3. Run Kilosort

4. Make spikets

5. Run analysis_code: with nidq.bin, dat file, kilosort output

   - ParseTJexperiment_NPX 
     : spikets from kilosort output, nFileSamp, syncON

     : markerts, pdOnTS, pdOffTS from nidq.bin
     : markervals from dat file




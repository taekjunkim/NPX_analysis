# NPX_analysis
Python code to analyze neuropixels data collected in Pasupathy Lab
<br>


1. NPX_parsing 
    - read raw data
    - extract trial information
    - draw a basic figure
    - organize data structure for future analysis
<br>

<img src="https://github.com/taekjunkim/NPX_analysis/blob/main/images/NPXparser.png" width="500">


2. Add_Analyze
   - NPX_parsing will create a json.gz or npz file for a specific task file.
   - Then, we can apply additional analyses based on the parsed file.
   - This folder is intended to collect together analyze code related to a specific task. 

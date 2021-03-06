## Download Datset
Dataset Link - https://archive.org/download/stackexchange

Data files to download:
- Dba - dba.stackexchange.com.7z
- Unix - unix.stackexchange.com.7z

Extract only the files Tags.xml and Posts.xml .

Put the two files of each of the 2 subsystems in the following data folders - data/dbaData and data/unixData respectively.


## PreProcessing - 
Run `python prepare_data.py `

to perform pre-processing over the data files and generate the required files for the experiment. You will need to change the parameter `DATA_DIR` in prepare_data.py to either data/dbaData or data/unixData depending the subsystem you are working wi
th. The output is generated in data/dbaData/output or data/unixData/output respectively.



## Experiment - 

In order to run the experiment you need to run the command - 
``` python main.py --data_dir=<data_dir> --start_seen=<start_seen> --end_seen=<end_seen> --plot_file=<plot_file> --measure= <centrality_measure>```
where,

`<data_dir>` - is the directory where data is present. It will be data/dbaData/output or data/unixData/output depending on the subsystem.

`<start_seen>` - Number of Seen Classes starting range

`<end_seen>` - Number of Seen Classes ending range

`<plot_file>` - name of the output plot (Precision @5 vs Number Seen Classes) that will be generated.

`<centrality_measure>` - name of the centrality measure to use 

The code reads the data and then creates a similarity_matrix using the boltzman machine. If the file similarity_matrix.npy is already present in data_dir, it skips its recomputation, if it is not present, it trains the similarity_matrix again and saves in the data_dir folder. It then runs the Active Zero Shot Learning Algorithm and gets the Precision @ 5 scores and produces the final plot and png file.



## Requirements - 

Python 3 (3.6)

Pickle

Numpy

Scipy

xml

Beautiful Soup

json

matplotlib

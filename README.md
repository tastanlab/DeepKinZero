# DeepKinZero
This repository contains the code and hyperparameters for the paper:
Iman Deznabi, Busra Arabaci, Mehmet Koyutürk, Oznur Tastan, DeepKinZero: Zero-Shot Learning for Predicting Kinase-Phosphosite Associations Involving Understudied Kinases, Bioinformatics, , btaa013, https://doi.org/10.1093/bioinformatics/btaa013

Please cite this paper if you use the code in this repository as part of a published research project.

## Abstract
Protein phosphorylation is a key regulator of protein function in signal transduction pathways. Kinases are the enzymes that catalyze the phosphorylation of other proteins in a target specific manner; dysregulation of phosphorylation is associated with many diseases including cancer.  Although advances in phosphoproteomics enable identification of phosphosites at the proteome level, most of the phosphoproteome is still in the dark:  more than 95% of reported human phosphosites have no known kinase. Determining which kinase is responsible for phoshorylating a site remains as an experimental challenge. Existing computational methods require several examples of known targets of a kinase to make accurate kinase specific predictions, yet for a large body of kinases no or only few target sites are reported. We present DeepKinZero, the first zero-shot learning approach to predict the kinase acting on a phosphosite for kinases with no known phosphosite information. DeepKinZero transfers knowledge from kinases with many known target phosphosites to those kinases with no known sites through a zero-shot learning model. The kinase specific positional amino acid preferences are learned using a  bidirectional recurrent network. Our computational experiments show that, as compared to baseline models, DeepKinZero achieves significant improvement in accuracy for kinases with no known phosphosites. By expanding  our knowledge on understudied kinases, DeepKinZero can help charting the phosphoproteome atlas.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
```
Python 3.6+
Tensorflow 1.9+
scikit-learn
numpy
```

### Installing

Just download the github repository
```
git clone https://github.com/tastanlab/DeepKinZero
```

### Run the trained model on your data
First run DownloadCheckpoint.py file to download the weights of the best model from Google Drive (you need tqdm, zipfile, requests libraries installed):
```
python3 DownloadCheckpoint.py
```
Then run the model on your test data and candidate set.
```
python3 predict.py -input path_to_your_data
```
You can also specify your Kinase candidates (the classes that the provided phosphorylation sites will be classified into) by providing the optional --candidates arguments for example:
```
python3 predict.py -input Data/PhosPhoELM/PhoELMdata.txt --candidates Data/AllCandidates.txt
```
By default the results will be written to Output/predictions.csv, you can change it by providing a path with argument --output, for other arguments and information please run:
```
python3 predict.py -h
```
Or:
```
python3 predict.py --help
```
### Input data format
The input data to predict.py should be tab delimeted and have 3 columns: site UNIPROT ID (eg. P07333), phosphosite residue and position (eg. Y561) and 15 neighboring residue of the phosphosite with phosphosite at center (eg. ESYEGNSYTFIDPTQ here the center Y is the phosphosite) while every line contains a phosphosite. For an example of such data please take a look at Data\PhosPhoELM\PhoELMdata.txt which contains pre-processed data from PhosphoELM dataset (http://phospho.elm.eu.org/). For train data for training your own model, the data should have 4 columns, so in addition to the 3 columns explained above it should also have the kinase UniprotID which phosphorylates the given phosphosite in that line (eg. Q05655). For an example of such data file please take a look at Data\Train_Phosphosite.txt which contains pre-processed data from Phosphosite plus dataset (https://www.phosphosite.org/). The candidate files should be single column file containing a single Kinase UniProtID (eg. O00506) in each line. For an example of such file please take a look at Data/AllCandidates.txt

## Train the model with your own data
For training the model with your own data please run:
```
python3 Train.py --TrainData path_to_your_training_data
```
Where the train data format is specified above, you can also provide validation data for tracking the performance of model training over time with argument --ValData and the candidates you want to consider with --ValKinaseCandidates. For more details please run:
```
python3 Train.py -h
```
Or:
```
python3 Train.py --help
```
For example you can train the model on data from Phosphosite plus dataset and validating on a selected portion of this data using the following command:
```
python3 Train.py --TrainData Data/Train_Phosphosite.txt --ValData Data/Val_Phosphosite_MultiLabel.txt --ValKinaseCandidates Data/Val_Candidate_Kinases.txt
```
## Authors
* **Iman Deznabi** - University of Massachusetts Amherst
* **Busra Arabaci** - Bilkent University
* **Mehmet Koyuturk** - Case Western Reserve University
* **Oznur Tastan** - Sabanci University

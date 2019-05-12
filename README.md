# DeepKinZero
This repository contains the code and hyperparameters for the paper:


Please cite this paper if you use the code in this repository as part of a published research project.

## Abstract
Protein phosphorylation is a key regulator of protein function in signal transduction pathways. Kinases are the enzymes that catalyze the phosphorylation of other proteins in a target specific manner; dysregulation of phosphorylation is associated with many diseases including cancer.  Although advances in phosphoproteomics enable identification of phosphosites at the proteome level, most of the phosphoproteome is still in the dark:  more than 95% of reported human phosphosites have no known kinase. Determining which kinase is responsible for phoshorylating a site remains as an experimental challenge. Existing computational methods require several examples of known targets of a kinase to make accurate kinase specific predictions, yet for a large body of kinases no or only few target sites are reported. We present DeepKinZero, the first zero-shot learning approach to predict the kinase acting on a phosphosite for kinases with no known phosphosite information. DeepKinZero transfers knowledge from kinases with many known target phosphosites to those kinases with no known sites through a zero-shot learning model. The kinase specific positional amino acid preferences are learned using a  bidirectional recurrent network. Our computational experiments show that, as compared to baseline models, DeepKinZero achieves significant improvement in accuracy for kinases with no known phosphosites. By expanding  our knowledge on understudied kinases, DeepKinZero can help charting the phosphoproteome atlas.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
```
* Python 3.6+
* Tensorflow 1.9+
* scikit-learn
* numpy
```

### Installing

Just download the github repository
```
git clone https://github.com/tastanlab/DeepKinZero
```

### Run the trained model on your data

```
python3 RunWithBestMethod.py --TestData path_to_your_data
```

### Input data format

## Train your own model

## Run with different parameters

## Authors

* **Iman Deznabi** - University of Massachussets Amherst
* **Busra Arabaci** - Bilkent University
* **Mehmet Koyuturk** - Case Western Reserve University
* **Oznur Tastan** - Sabanci University

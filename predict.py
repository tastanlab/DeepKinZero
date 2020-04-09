# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 18:08:52 2019

@author: Iman
"""
from deepkinzero_EndToEnd import Run
import argparse
import os

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--BestModelCheckpoint', help='The path for the checkpoint folder of the best model', type=str, default='BestModelCheckpoint')
    # Input Paths
    parser.add_argument('-input', help='The path for your input data, \n \
                        data file should be tab delimeted and have 3 columns: \n \
                        site UNIPROT ID (eg. P07333), phosphosite residue and position (eg. Y561) and 15 neighboring residue of the phosphosite (eg. ESYEGNSYTFIDPTQ here the center Y is the phosphosite) \n \
                        Check Data\\PhosPhoELM\\PhoELMdata.txt for an example', type=str, required=True)
    parser.add_argument('--candidates', help='The path to the file which contains the list of kinase candidates, \n \
                        these are your potential kinases which can phosphorylate your phosphosites', type=str, default='Data/AllCandidates.txt')
    parser.add_argument('--output', help='Path to where the predicted kinases will be saved', type=str, default='Output/predictions.txt')
    parser.add_argument('--NumOfTop', help='Number of top kinases (highest predicted probability) to include in results', type=int, default=10)
    
    args=parser.parse_args()
    # Its just for initializing the right model please don't change here
    ModelParams = {"rnn_unit_type": "LNlstm", "num_layers": 2, "num_hidden_units": 512, "dropoutval": 0.5, "learningrate": 0.001, "useAtt": True, "useEmbeddingLayer": False, "useEmbeddingLayer": False, "num_of_Convs": [], "UseBatchNormalization1": True, "UseBatchNormalization2": True, "EMBEDDING_DIM": 500, "ATTENTION_SIZE": 20, "IncreaseEmbSize": 0, "Bidirectional":True, "Dropout1": True, "Dropout2": True, "Dropout3": False, "regs": 0.001, "batch_size": 64, "ClippingGradients": 9.0, "activation1": None, "LRDecay":True, "seed":100, "NumofModels": 10} #a dictionary indicating the parameters provided for the model
    
    Run(Model = 'ZSL', TrainingEpochs = 50,
        AminoAcidProperties = False, ProtVec = True, NormalizeDE=True,
        ModelParams= ModelParams, Family = True, Group = True, Pathways = False, Kin2Vec=True, Enzymes = True,
        LoadModel = True, CustomLabel="RunWithBestModel",
        TrainData = '', TestData = args.input, ValData='', TestKinaseCandidates= args.candidates, ValKinaseCandidates= '',
        ParentLogDir = 'Logs', EmbeddingOrParams=True, OutPath = args.output, Top_n = args.NumOfTop, CheckpointPath=args.BestModelCheckpoint)
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 14:01:53 2019

@author: Iman
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 21:24:57 2018

@author: Iman
"""
from deepkinzero_EndToEnd import Run
import argparse
import os

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--ModelEpochs', help='Number of Epochs for training the main model', default=50, type=int)
    
    # Get data paths
    parser.add_argument('--TrainData', help='The path for train data', type=str, default='Data/Train_Phosphosite.txt')
    # Val data Paths
    parser.add_argument('--ValData', help='The path for validation data, leave empty for not running on validation data', type=str, default='Data/Val_Phosphosite_MultiLabel.txt')
    parser.add_argument('--ValKinaseCandidates', help='The path to the file which contains the list of validation candidates', type=str, default='Data/Val_Candidate_Kinases.txt')
        
    parser.add_argument('--NumofModels', help='Number of models to ensemble', type=int, default=10)
    
    parser.add_argument('--LoadModel', help='if provided the model will be loaded from last checkpoint', action='store_true')
    
    args=parser.parse_args()
    print(args)
    
    ModelParams = {"rnn_unit_type": "LNlstm", "num_layers": 2, "num_hidden_units": 512, "dropoutval": 0.5, "learningrate": 0.001, "useAtt": True, "useEmbeddingLayer": False, "useEmbeddingLayer": False, "num_of_Convs": [], "UseBatchNormalization1": True, "UseBatchNormalization2": True, "EMBEDDING_DIM": 500, "ATTENTION_SIZE": 20, "IncreaseEmbSize": 0, "Bidirectional":True, "Dropout1": True, "Dropout2": True, "Dropout3": False, "regs": 0.001, "batch_size": 64, "ClippingGradients": 9.0, "activation1": None, "LRDecay":True, "seed":100, "NumofModels": args.NumofModels} #a dictionary indicating the parameters provided for the model
    
    print(ModelParams)
    
    paramsStr = str(ModelParams["rnn_unit_type"]) + "," + str(ModelParams["num_layers"]) + "," + str(ModelParams["num_hidden_units"]) + "," +  str(ModelParams["dropoutval"]) + "," + str(ModelParams["learningrate"]) + "," + str(ModelParams["useAtt"]) + "," + \
                str(ModelParams["useEmbeddingLayer"]) + "," + str(ModelParams["num_of_Convs"]).replace(",","_") + \
                "," + str(ModelParams["UseBatchNormalization1"]) + "," +  str(ModelParams["UseBatchNormalization2"]) + "," + str(ModelParams["EMBEDDING_DIM"]) + "," + str(ModelParams["ATTENTION_SIZE"]) + "," + \
                str(ModelParams["IncreaseEmbSize"]) + "," + str(ModelParams["Bidirectional"]) + "," +  str(ModelParams["Dropout1"]) + "," +  str(ModelParams["Dropout2"]) + "," + str(ModelParams["Dropout3"]) + "," + \
                str(ModelParams["regs"]) + "," + str(ModelParams["batch_size"]) + "," +  str(ModelParams["ClippingGradients"]) + "," + str(ModelParams["activation1"]) + "," + str(ModelParams["LRDecay"]) + "," + str(args.ModelEpochs) + "," + str(ModelParams["seed"])
    if not os.path.isfile('TrainResults.csv'):
        with open('TrainResults.csv','w+') as outfile:
            print("rnn_unit_type, num_layers, num_hidden_units, dropoutval, learningrate, useAtt, \
                  useEmbeddingLayer, num_of_Convs, UseBatchNormalization1, UseBatchNormalization2, EMBEDDING_DIM, ATTENTION_SIZE, IncreaseEmbSize, Bidirectional, Dropout1, Dropout2, Dropout3, \
                  regs, batch_size, ClippingGradients, activation1, LRDecay, ModelEpochs, seed, Train Accuracy, Loss, Validation Loss, Validation Accuracy, Top3Acc, Top5Acc, Top10Acc", file=outfile)
    with open('TrainResults.csv','a+') as outfile:
        Train_Evaluations, Val_Evaluations, ValKinUniProtIDs = Run(Model = 'ZSL', TrainingEpochs = args.ModelEpochs,
            AminoAcidProperties = False, ProtVec = True, NormalizeDE=True,
            ModelParams= ModelParams, Family = True, Group = True, Pathways = False, Kin2Vec=True, Enzymes = True,
            LoadModel = args.LoadModel, CustomLabel="Train",
            TrainData = args.TrainData, ValData=args.ValData, ValKinaseCandidates= args.ValKinaseCandidates, ParentLogDir = 'Logs')
        
     
        print(paramsStr + "," + str(Train_Evaluations["Accuracy"]) + "," + str(Train_Evaluations["Loss"]) + "," + str(Val_Evaluations["Loss"]) + "," + str(Val_Evaluations["Accuracy"])  + "," + str(Val_Evaluations["Top3Acc"]) + "," + str(Val_Evaluations["Top5Acc"]) + "," + str(Val_Evaluations["Top10Acc"]), file=outfile)
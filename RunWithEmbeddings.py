# -*- coding: utf-8 -*-

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
from random import *
import sys, os
import tensorflow as tf

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    # The Zero-Shot learning model
    parser.add_argument('--ModelEpochs', help='Number of Epochs for training the main model', default=50, type=int)
    parser.add_argument('--NumofModels', help='Number of models to ensemble', type=int, default=10)
    
    # Get data paths
    parser.add_argument('--TrainData', help='The path for train data', type=str, default='Data/Train_Phosphosite_new.txt')
    #parser.add_argument('--TestData', help='The path for test data', type=str, default='Data/Test_Phosphosite_MultiLabel.txt')
    # Test data Paths
    parser.add_argument('--TestData', help='The path for test data', type=str, default='Data/Test_Phosphosite_MultiLabel.txt')
    parser.add_argument('--TestKinaseCandidates', help='The path to the file which contains the list of test candidates', type=str, default='Data/Candidate_Kinases_new.txt')
    # Val data Paths
    parser.add_argument('--ValData', help='The path for validation data', type=str, default='Data/Val_Phosphosite_MultiLabel.txt')
    parser.add_argument('--ValKinaseCandidates', help='The path to the file which contains the list of test candidates', type=str, default='Data/Val_Candidate_Kinases.txt')
    
    parser.add_argument('--RandomSeed', help='Random Seed', type=int, default=100)
    
    # Kinase Embedding options
    parser.add_argument('--UseFamily', action='store_true', help='Use Kinase Family in Class Embedding')
    parser.add_argument('--UseGroup', action='store_true', help='Use Kinase Group in Class Embedding')
    parser.add_argument('--UsePathway', action='store_true', help='Use KEGG Pathways as class embedding')
    parser.add_argument('--UseEnzymes', action='store_true', help='Use Enzymes vectors as class embedding')
    parser.add_argument('--UseKin2Vec', action='store_true', help='Use Vectors generated from protein2vec for Kinase Sequences as class embedding')
    
    parser.add_argument('--DoNotNormalizeDE', dest="NormalizeDE", action='store_false', default=True, help='Do not normalize the data embedding vectors')
    
    # Data Embedding types
    parser.add_argument('--AminoAcidProperties', action='store_true', help='Use other properties for Amino Acids while embedding Options are True or False')
    parser.add_argument('--UseProtVec', action='store_true', help='Use ProtVec to convert the dataembedding sequence to a vector')
    
    parser.add_argument('--LR', help='Learning Rate', type=float, default=0.001)
    
    parser.add_argument('--LoadModel', action='store_true', help='Load previous methods')
    
    args=parser.parse_args()
    ModelParams = {"rnn_unit_type": "LNlstm", "num_layers": 2, "num_hidden_units": 512, "dropoutval": 0.5, "learningrate": args.LR, "useAtt": True, "useEmbeddingLayer": False, "useEmbeddingLayer": False, "num_of_Convs": [], "UseBatchNormalization1": True, "UseBatchNormalization2": True, "EMBEDDING_DIM": 500, "ATTENTION_SIZE": 20, "IncreaseEmbSize": 0, "Bidirectional":True, "Dropout1": True, "Dropout2": True, "Dropout3": False, "regs": 0.001, "batch_size": 64, "ClippingGradients": 9.0, "activation1": None, "LRDecay":True, "seed":args.RandomSeed, "NumofModels": args.NumofModels} #a dictionary indicating the parameters provided for the model
    
    args=parser.parse_args()
    print(args)
    
    
    print(ModelParams)
    
    paramsStr = str(str(args.UseFamily) + "," + str(args.UseGroup)+ "," + str(args.UsePathway)+ "," + str(args.UseEnzymes)+ "," + str(args.UseKin2Vec) + "," + str(args.AminoAcidProperties) + "," + str(args.UseProtVec) + ","  + str(ModelParams["seed"]))
    if not os.path.isfile('RunWithEmbeddings_cropcost1.csv'):
        with open('RunWithEmbeddings_cropcost1.csv','w+') as outfile:
            print("UseFamily,UseGroup,UsePathway,UseEnzymes,UseKin2Vec,AminoAcidProperties,UseProtVec,seed,Train_accuracy,Val_accuracy,test_accuracy,Train_Loss,Val_loss,test_loss,top3Accuracy_Val,top5Accuracy_Val,top10Accuracy_Val,top3Accuracy_test,top5Accuracy_test,top10Accuracy_test", file=outfile)
        #if os.stat('RunWithEmbeddings.csv').st_size == 0:
        #        print("UseFamily,UseGroup,UsePathway,UseEnzymes,UseKin2Vec,AminoAcidProperties,UseProtVec,seed,", file=outfile)
        #try:
    Train_Evaluations, Val_Evaluations, Test_Evaluations = Run(Model = 'ZSL', TrainingEpochs = args.ModelEpochs,
        AminoAcidProperties = args.AminoAcidProperties, ProtVec = args.UseProtVec, NormalizeDE=args.NormalizeDE,
        ModelParams= ModelParams, Family = args.UseFamily, Group = args.UseGroup, Pathways = args.UsePathway, Kin2Vec=args.UseKin2Vec, Enzymes = args.UseEnzymes,
        LoadModel = args.LoadModel, CustomLabel="RunWithEmbeddings",
        TrainData = args.TrainData, TestData = args.TestData, ValData=args.ValData, TestKinaseCandidates= args.TestKinaseCandidates, ValKinaseCandidates= args.ValKinaseCandidates,
        ParentLogDir = 'RunWithEmbeddings_cropcost1', EmbeddingOrParams=True)
        
    with open('RunWithEmbeddings_cropcost1.csv','a+') as outfile:        
        print(paramsStr + "," + str(Train_Evaluations["Accuracy"]) + "," + str(Val_Evaluations["Accuracy"]) + "," + str(Test_Evaluations["Accuracy"]) + 
              "," + str(Train_Evaluations["Loss"]) + "," + str(Val_Evaluations["Loss"]) + "," + str(Test_Evaluations["Loss"]) + 
              "," + str(Val_Evaluations["Top3Acc"]) + "," + str(Val_Evaluations["Top5Acc"]) + "," + str(Val_Evaluations["Top10Acc"]) + 
              "," + str(Test_Evaluations["Top3Acc"]) + "," + str(Test_Evaluations["Top5Acc"]) + "," + str(Test_Evaluations["Top10Acc"]), file=outfile)
        #except tf.errors.InvalidArgumentError as IAE:
        #    print(paramsStr + ',InvalidArgumentError,' + str(sys.exc_info()), file=outfile)
        #except:
        #    print(paramsStr + ",UnknownError," + str(sys.exc_info()), file=outfile)
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 18:08:52 2019

@author: Iman
"""
import datetime
from deepkinzero_EndToEnd import Run
import argparse
from random import *
import sys, os
import tensorflow as tf

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--BestModelCheckpoint', help='The path for the checkpoint folder of the best model', type=str, default='BestModelCheckpoint')
    # Get data paths
    parser.add_argument('--TrainData', help='The path for train data', type=str, default='Data/Train_Phosphosite_new.txt')
    #parser.add_argument('--TestData', help='The path for test data', type=str, default='Data/Test_Phosphosite_MultiLabel.txt')
    # Test data Paths
    parser.add_argument('--TestData', help='The path for test data', type=str, default='Data/PhosphoELM/test_pelm/Test_PhoELM_MultiLabel.txt')
    parser.add_argument('--TestKinaseCandidates', help='The path to the file which contains the list of test candidates', type=str, default='Data/PhosphoELM/test_pelm/Candidate_Kinases_pELM.txt')
    # Val data Paths
    parser.add_argument('--ValData', help='The path for validation data', type=str, default='Data/Val_Phosphosite_MultiLabel.txt')
    parser.add_argument('--ValKinaseCandidates', help='The path to the file which contains the list of test candidates', type=str, default='Data/Val_Candidate_Kinases.txt')
            
    args=parser.parse_args()
    ModelParams = {"rnn_unit_type": "LNlstm", "num_layers": 2, "num_hidden_units": 512, "dropoutval": 0.5, "learningrate": 0.001, "useAtt": True, "useEmbeddingLayer": False, "useEmbeddingLayer": False, "num_of_Convs": [], "UseBatchNormalization1": True, "UseBatchNormalization2": True, "EMBEDDING_DIM": 500, "ATTENTION_SIZE": 20, "IncreaseEmbSize": 0, "Bidirectional":True, "Dropout1": True, "Dropout2": True, "Dropout3": False, "regs": 0.001, "batch_size": 64, "ClippingGradients": 9.0, "activation1": None, "LRDecay":True, "seed":100, "NumofModels": 10} #a dictionary indicating the parameters provided for the model
    
    args=parser.parse_args()
    print(args)
    
    
    print(ModelParams)
    
    #paramsStr = str(str(args.UseFamily) + "," + str(args.UseGroup)+ "," + str(args.UsePathway)+ "," + str(args.UseEnzymes)+ "," + str(args.UseKin2Vec) + "," + str(args.AminoAcidProperties) + "," + str(args.UseProtVec) + ","  + str(ModelParams["seed"]))
    if not os.path.isfile('RunWithBestMethod.csv'):
        with open('RunWithBestMethod.csv','w+') as outfile:
            print("Time,Dataset,Train_accuracy,Val_accuracy,test_accuracy,Train_Loss,Val_loss,test_loss,top3Accuracy_Val,top5Accuracy_Val,top10Accuracy_Val,top3Accuracy_test,top5Accuracy_test,top10Accuracy_test", file=outfile)
        #if os.stat('RunWithEmbeddings.csv').st_size == 0:
        #        print("UseFamily,UseGroup,UsePathway,UseEnzymes,UseKin2Vec,AminoAcidProperties,UseProtVec,seed,", file=outfile)
        #try:
    Train_Evaluations, Val_Evaluations, Test_Evaluations = Run(Model = 'ZSL', TrainingEpochs = 50,
        AminoAcidProperties = False, ProtVec = True, NormalizeDE=True,
        ModelParams= ModelParams, Family = True, Group = True, Pathways = False, Kin2Vec=True, Enzymes = True,
        LoadModel = True, CustomLabel="RunWithBestModel",
        TrainData = args.TrainData, TestData = args.TestData, ValData=args.ValData, TestKinaseCandidates= args.TestKinaseCandidates, ValKinaseCandidates= args.ValKinaseCandidates,
        ParentLogDir = 'RunWithBestModel', EmbeddingOrParams=True, CheckpointPath=args.BestModelCheckpoint)
        
    with open('RunWithBestMethod.csv','a+') as outfile:        
        print(str(datetime.datetime.now()) + "," + args.TestData + "," + str(Train_Evaluations["Accuracy"]) + "," + str(Val_Evaluations["Accuracy"]) + "," + str(Test_Evaluations["Accuracy"]) + 
              "," + str(Train_Evaluations["Loss"]) + "," + str(Val_Evaluations["Loss"]) + "," + str(Test_Evaluations["Loss"]) + 
              "," + str(Val_Evaluations["Top3Acc"]) + "," + str(Val_Evaluations["Top5Acc"]) + "," + str(Val_Evaluations["Top10Acc"]) + 
              "," + str(Test_Evaluations["Top3Acc"]) + "," + str(Test_Evaluations["Top5Acc"]) + "," + str(Test_Evaluations["Top10Acc"]), file=outfile)
        #except tf.errors.InvalidArgumentError as IAE:
        #    print(paramsStr + ',InvalidArgumentError,' + str(sys.exc_info()), file=outfile)
        #except:
        #    print(paramsStr + ",UnknownError," + str(sys.exc_info()), file=outfile)
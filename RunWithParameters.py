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
import sys
import tensorflow as tf

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    # The Zero-Shot learning model
    parser.add_argument('--ModelEpochs', help='Number of Epochs for training the main model', default=50, type=int)
    
    # Get data paths
    parser.add_argument('--TrainData', help='The path for train data', type=str, default='Data/Train_Phosphosite_new.txt')
    #parser.add_argument('--TestData', help='The path for test data', type=str, default='Data/Test_Phosphosite_MultiLabel.txt')
    # Test data Paths
    parser.add_argument('--TestData', help='The path for test data', type=str, default='Data/Test_Phosphosite_MultiLabel.txt')
    parser.add_argument('--TestKinaseCandidates', help='The path to the file which contains the list of test candidates', type=str, default='Data/Candidate_Kinases_new.txt')
    # Val data Paths
    parser.add_argument('--ValData', help='The path for validation data', type=str, default='Data/Val_Phosphosite_MultiLabel.txt')
    parser.add_argument('--ValKinaseCandidates', help='The path to the file which contains the list of test candidates', type=str, default='Data/Val_Candidate_Kinases.txt')
    
    parser.add_argument('--rnn_unit_type', help='RNN unit type to use', type=str, default='LNlstm')
    parser.add_argument('--num_layers', help='Number of Layers to try', type=int, default=2)
    parser.add_argument('--num_hidden_units', help='Number of nodes for each layer', type=int, default=512)
    parser.add_argument('--dropoutval', help='dropout values to try', type=float, default=0.5)
    parser.add_argument('--learningrate', help='Learning Rates to try', type=float, default=0.001)
    parser.add_argument('--useAtt', help='Use attention?', action='store_true')
    parser.add_argument('--num_of_Convs', help='Num of convs to try', nargs='*', type=int, default=[])
    parser.add_argument('--no-BatchNormalization1', dest='UseBatchNormalization1', action='store_false')
    parser.add_argument('--no-BatchNormalization2', dest='UseBatchNormalization2', help='-Should we use batchnorm at second place', action='store_false')
    parser.add_argument('--ATTENTION_SIZE', help='Attention sizes to try', type=int, default=5)
    parser.add_argument('--no-Bidirectional', help='Should we use bidirectional lstm?', dest='Bidirectional', action='store_false')
    parser.add_argument('--no-Dropout1', help='Should we use dropout at first', dest='Dropout1', action='store_false')
    parser.add_argument('--no-Dropout2', help='Should we use dropout at second', dest='Dropout2', action='store_false')
    #parser.add_argument('--Dropout3', help='Number of filters for each Conv Layers to use in BiRNN before BiRNN model leave empty to not use Conv layers', nargs='*', type=int, default=[])
    parser.add_argument('--regs', help='regularization values to try', type=float, default=0)
    parser.add_argument('--batch_size', help='Batch sizes to try', type=int, default=64)
    parser.add_argument('--ClippingGradients', help='Clipping gradient values to try', type=float, default=9.0)
    parser.add_argument('--activation1', help='activation for intermediate layers of RNN', type=str, default='None')
    
    parser.add_argument('--seed', help='Random Seed', type=int, default=100)
    
    parser.add_argument('--no-Decay', help='Do not decay the learning rate', dest='LRDecay', action='store_false')
    
    parser.add_argument('--NumofModels', help='Number of models to ensemble', type=int, default=10)
    
    parser.add_argument('--no-NormalizeData', help='Normalize data', dest='NormalizeData', action='store_false')
    
    parser.add_argument('--LoadModel', help='if provided the model will be loaded from last checkpoint', action='store_true')
    
    args=parser.parse_args()
    print(args)
    
    if args.activation1 == 'None':
        activation = None
    else:
        activation = args.activation1
    seed = args.seed
    if seed == 0:
        seed = None
    ModelParams = {"rnn_unit_type": args.rnn_unit_type, "num_layers": args.num_layers, "num_hidden_units": args.num_hidden_units, "dropoutval": args.dropoutval, "learningrate": args.learningrate, 
                   "useAtt": args.useAtt, "useEmbeddingLayer": False, "num_of_Convs": args.num_of_Convs, "UseBatchNormalization1": args.UseBatchNormalization1, 
                   "UseBatchNormalization2": args.UseBatchNormalization2, "EMBEDDING_DIM": 100, "ATTENTION_SIZE": args.ATTENTION_SIZE, "IncreaseEmbSize": 0, "Bidirectional":args.Bidirectional, "Dropout1": args.Dropout1
                   , "Dropout2": args.Dropout2, "Dropout3": False, "regs": args.regs, "batch_size": args.batch_size, "ClippingGradients": args.ClippingGradients, "activation1":activation, "LRDecay": args.LRDecay, "seed":seed,
                   "NumofModels": args.NumofModels} #a dictionary indicating the parameters provided for the model
    
    print(ModelParams)
    
    paramsStr = str(ModelParams["rnn_unit_type"]) + "," + str(ModelParams["num_layers"]) + "," + str(ModelParams["num_hidden_units"]) + "," +  str(ModelParams["dropoutval"]) + "," + str(ModelParams["learningrate"]) + "," + str(ModelParams["useAtt"]) + "," + \
                str(ModelParams["useEmbeddingLayer"]) + "," + str(ModelParams["useEmbeddingLayer"]) + "," + str(ModelParams["num_of_Convs"]).replace(",","_") + \
                "," + str(ModelParams["UseBatchNormalization1"]) + "," +  str(ModelParams["UseBatchNormalization2"]) + "," + str(ModelParams["EMBEDDING_DIM"]) + "," + str(ModelParams["ATTENTION_SIZE"]) + "," + \
                str(ModelParams["IncreaseEmbSize"]) + "," + str(ModelParams["Bidirectional"]) + "," +  str(ModelParams["Dropout1"]) + "," +  str(ModelParams["Dropout2"]) + "," + str(ModelParams["Dropout3"]) + "," + \
                str(ModelParams["regs"]) + "," + str(ModelParams["batch_size"]) + "," +  str(ModelParams["ClippingGradients"]) + "," + str(args.NormalizeData) + "," + str(ModelParams["activation1"]) + "," + str(ModelParams["LRDecay"]) + "," + str(args.ModelEpochs) + "," + str(ModelParams["seed"])
            
    with open('RunWithParameters4.csv','a+') as outfile:
        #try:
        Train_Loss, Train_accuracy, Val_loss, Val_accuracy, top5Accuracy_Val, top10Accuracy_Val, test_loss, test_accuracy, top5Accuracy_test, top10Accuracy_test = Run(Model = 'ZSL', TrainingEpochs = args.ModelEpochs,
            AminoAcidProperties = False, ProtVec = True, NormalizeDE=args.NormalizeData,
            ModelParams= ModelParams, Family = True, Group = True, Pathways = False, Kin2Vec=True, Enzymes = True,
            LoadModel = args.LoadModel, CustomLabel="hyperparamether",
            TrainData = args.TrainData, TestData = args.TestData, ValData=args.ValData, TestKinaseCandidates= args.TestKinaseCandidates, ValKinaseCandidates= args.ValKinaseCandidates,
            ParentLogDir = 'RunWithParameters3')
        
        
        print(paramsStr + "," + str(Train_accuracy) + "," + str(Val_accuracy) + "," + str(test_accuracy) + "," + str(Train_Loss) + "," + str(Val_loss) + "," + str(test_loss) + "," + str(top5Accuracy_Val) + "," + str(top10Accuracy_Val) + "," + str(top5Accuracy_test)+ "," + str(top10Accuracy_Val), file=outfile)
        #except tf.errors.InvalidArgumentError as IAE:
        #    print(paramsStr + ',InvalidArgumentError,' + str(sys.exc_info()), file=outfile)
        #except:
        #    print(paramsStr + ",UnknownError," + str(sys.exc_info()), file=outfile)
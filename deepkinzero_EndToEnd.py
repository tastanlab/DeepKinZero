""" Main Code to Run the model
This file includes the necessary methods to run the model with different parameters

Example:
    An example of running this file is given below:
        $ python3 Run.py --UseGPU --Embedding=BiRNN --UseFocus --UseFamily --UseGroup --UseType --UsePathway --DataEmbeddingEpochs=65 --ModelEpochs=1000 >log.txt
        $ python Run.py --Embedding=BiRNN --UseFocus --UseFamily --UseGroup --UseType --UsePathway --UseGene2Vec --UseKin2Vec
        $ python Run.py --Model=BiRNN --ParameterSelection
        $ python Run.py --Embedding=BiRNN --UseFamily --UseGroup --UseType --DoNotDoKfold --EmbeddingDIM=200 --DataEmbeddingInitEpochs=50 --ModelInitEpochs=10 --ModelEachEpochs=10 --ModelTimes=3000 --RNNNodeType=None --NumOfConvLayers 400
        $ python3 Run.py --Embedding=BiRNN --UseFamily --UseGroup --UseType --DoNotDoKfold --EmbeddingDIM=400 --DEhiddenUnits=400 --DataEmbeddingInitEpochs=500 --ModelInitEpochs=10 --ModelEachEpochs=10 --ModelTimes=1000 --RNNNodeType=LNlstm
        $ python3 Run.py --Embedding=BiRNN --UseGene2Vec --UseGroup --UseFamily --UseGOC --UseGOP --DoNotDoKfold --EmbeddingDIM=500 --DEhiddenUnits=500 --DataEmbeddingInitEpochs=50 --ModelInitEpochs=1000 --LoadDEModel --ModelEachEpochs=0 --ModelTimes=0 --RNNNodeType=LNlstm
        $ python3 Run.py --Embedding=BiRNN --UseGene2Vec --UseGroup --UseFamily --UseKin2Vec --UseEnzymes --DoNotDoKfold --EmbeddingDIM=500 --DEhiddenUnits=500 --DataEmbeddingInitEpochs=50 --ModelInitEpochs=1000 --LoadDEModel --ModelEachEpochs=0 --ModelTimes=0 --RNNNodeType=LNlstm
    You can get the available arguments by runnung python Run.py -h
"""
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore")

import os, argparse, time
import datetime
import sys

import numpy as np
import tensorflow as tf

#Data Utilities
from datautils import (dataset, KinaseEmbedding)

#Models for Zero Shot Learning
#from SimpleZSLTF import SimpleZSLTF
#from KNearestZSL import KNearestZSL

#Models for Data Embedding
#from BiRNNModel import BiRNNModel
from EndToEnd import EndToEndModel

from EndToEnd import GetAccuracyMultiLabel

from EndToEnd import ensemble

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#Evaluations
#from Evaluations import (KfoldWithZSL, KfoldWithZSLRNN, KfoldTest, RareAccuracy, ZSLGivenModel, RunOverTrainandTest, KfoldwithZSLGivenModel, FindBestEmbedding)
#from EvaluationMetrics import (get_top_n_accuracy, get_f1_score, get_conf_matrix, get_calcs_for_class, get_accuracy, get_prec_macro, get_prec_micro, get_recall_macro, get_recall_micro, get_precision, get_recall, get_micro_avg_precision, get_macro_avg_precision, get_auc, get_average_precision, get_map, getClassificationReport)

from sklearn.metrics import accuracy_score

def FindTrueClassIndices(ClassEmbeddings, CandidateKinases, Multilabel = False):
    """
    Find the indices of the true classes in the candidate class embeddings
    Args:
        ClassEmbeddings: The class embeddings to find their indices
        CandidateKinases: The list of candidate kinases
    
    Returns:
        a list of index per entry in ClassEmbeddings showing its index in CandidateKinases
    """
    
    TrueClassIDX = []
    for CEW1 in ClassEmbeddings:
        if not Multilabel:
            idx = np.where(np.all(np.equal(CEW1, CandidateKinases), axis=1))
            TrueClassIDX.append(idx[0][0])
        else:
            indices = []
            for CE in CEW1:
                idx = np.where(np.all(np.equal(CE, CandidateKinases), axis=1))
                indices.append(idx[0][0])
            TrueClassIDX.append(indices)
    TrueClassIDX = np.array(TrueClassIDX)
    return TrueClassIDX

def GetResultsMetrics( Accuracy, Confmat, Probabilities, Y_True, Y_pred, Target_names):
    """ Returns Percision, Recall, fscore micro and fscore macro, top3, top5, top10 accuracies, MAP, AUC and creates a classification report.
    Args:
        Accuracy (float): accuracy of the model
        Confmat (C * C matrix of ints): Confusion matrix of the 
    Returns:
        Accuracy (float)
        Percision (float)
        Recall (float)
        fscoremicro (float)
        fscoremacro (float)
        top3 (float): top3 accuracy of the model
        top5 (float): top5 accuracy of the model
        top10 (float): top10 accuracy of the model
        MAP (float): mean average percision of the model
        AUC (float): area under the curve of the model
        CR: Classification report
    """
    #topnaccuracy = get_top_n_accuracy(5, Probabilities, Y_true)
    #f1 = get_f1_score(Probabilities, realclassstr, classstr)
    Percision = get_prec_micro(Y_True, Y_pred)
    Recall = get_recall_micro(Y_True, Y_pred)
    fscoremicro, fscoremacro = get_f1_score(Probabilities, Y_True, Y_pred)
    top3 = get_top_n_accuracy(3, Probabilities, Y_True)
    top5 = get_top_n_accuracy(5, Probabilities, Y_True)
    top10 = get_top_n_accuracy(10, Probabilities, Y_True)
    MAP = get_map(Probabilities, Y_True, Y_pred)
    AUC = get_auc(Probabilities, Y_True, Y_pred)
    CR = getClassificationReport(Probabilities, Y_True, Y_pred, Target_names)
    #print("Top_n_Accuracy : ",topnaccuracy, "f1 : ", f1)
    return Accuracy , Percision , Recall , fscoremicro , fscoremacro , top3 , top5 , top10 , MAP, AUC, CR


def getStrParam(Params):
    out = "UT={}_NL={}_NH={}_DO={}_LR={}_At={}_AtS={}_BN1={}_BN2={}_DO1={}_DO2={}_Regs={}_LRD={}".format(
            Params["rnn_unit_type"], Params["num_layers"], Params["num_hidden_units"], Params["dropoutval"], Params["learningrate"], Params["useAtt"], Params["ATTENTION_SIZE"], Params["UseBatchNormalization1"], Params["UseBatchNormalization2"], Params["Dropout1"], Params["Dropout2"], Params["regs"], Params["LRDecay"])
    
    return out
def createFolderName(OtherAlphabets, Gene2Vec, Family, Group, Pathways, Kin2Vec, Enzymes, Params, MainModel, EmbeddingOrParams=False):
    """ Creates two strings representing the parameters given for storing logs and results one of the strings is the name of the folder for ZSL results and the other one for saving data embedding models logs and models
    Args:
        OtherAlphabets (bool): Use other properties of Amino Acids
        Gene2Vec (bool): Use Gene2Vec for converting protein sequences into vector of continuos numbers
        RNNfeatures (bool): Use RNN as DataEmbedding tool
        BiRNNfeatures (bool): Use BiRNN as DataEmbedding tool
        BiRNNfocusfeatures (bool): Use attention mechanism in BiRNN
        Family (bool): Use Kinase family data in Class Embeddings
        Group (bool): Use Kinase group (superfamily) data in Class Embeddings
        STY (bool): Use Kinase type data (S T or Y) in Class Embeddings
        Pathways (bool): Use KEGG pathways data in Class Embeddings
        ConvLayers (int array): Number of Convolutional Layers to use in BiRNN
        MainModel (str): What is the main model for classification (it can be SZSL, SVM, LogisticRegression, RNN, BiRNN)
        
    Returns:
        Name (str): The name of the folder for ZSL results
        DEFolder (str): The name of the folder for Data Embedding model logs and checkpoints
    """
    DEFolder = ""
    if EmbeddingOrParams:
        Name = MainModel
        if Params["Bidirectional"]:
            Name = Name + "BiRNN-"
            DEFolder = DEFolder + "BiRNN-"
        if Params["useAtt"]:
            Name = Name + "Att-"
            DEFolder = DEFolder + "Att-"
        if OtherAlphabets:
            Name = Name + "OA-"
            DEFolder = DEFolder + "OA-"
        if Gene2Vec:
            Name = Name + "G2V-"
            DEFolder = DEFolder + "G2V-"
        if Family:
            Name = Name + "Fam-"
        if Group:
            Name =  Name + "Group-"
        if Pathways:
            Name = Name + "Path-"
        if Kin2Vec:
            Name = Name + "K2V-"
        if Enzymes:
            Name = Name + "Enz-"
        if len(Params["num_of_Convs"])>0:
            Name= Name + str(Params["ConvLayers"]) + "Conv"
        return Name, DEFolder
    else:
        return getStrParam(Params), DEFolder
    #now = datetime.datetime.now()
    #return Name+now.strftime("%Y-%m-%d%H_%M"), DEFolder
    
def Run(Model = "ZSL",
        AminoAcidProperties = False, ProtVec = False, NormalizeDE = True,
        ModelParams = None, Family = False, Group = False, Pathways = False, Kin2Vec=False, Enzymes = True,
        TrainingEpochs = 70, LoadModel = False,
        CustomLabel="", seed=None,
        TrainData = '', 
        TestData = '', TestKinaseCandidates= '', 
        ValData='', ValKinaseCandidates= '',
        ParentLogDir='Logs', EmbeddingOrParams=False, CheckpointPath=None):
    """ This is the main function which is responsible with running the whole code given the parameters
    Args:
        AminoAcidProperties (bool): Use other properties of Amino Acids
        ProtVec (bool): Use ProtVec for converting protein sequences into vector of continuos numbers
        RNNfeatures (bool): Use RNN as DataEmbedding tool
        BiRNNfeatures (bool): Use BiRNN as DataEmbedding tool
        BiRNNfocusfeatures (bool): Use attention mechanism in BiRNN
        Family (bool): Use Kinase family data in Class Embeddings
        Group (bool): Use Kinase group (superfamily) data in Class Embeddings
        STY (bool): Use Kinase type data (S T or Y) in Class Embeddings
        Pathways (bool): Use KEGG pathways data in Class Embeddings
        Kin2Vec (bool): Use Vectors generated from protein2vec for Kinase Sequences in Class Embeddings
        MainModel (str): What is the main model for classification (it can be SZSL, SVM, LogisticRegression, RNN, BiRNN)
    """
    # Create a folder name based on paramethers
    FolderName, DEFolder = createFolderName(AminoAcidProperties, ProtVec, Family, Group, Pathways, Kin2Vec, Enzymes, ModelParams, Model, EmbeddingOrParams)
    FolderName = os.path.join(ParentLogDir, FolderName)
    i = 1
    while os.path.exists(FolderName + "_" + str(i)):
        i += 1
    if LoadModel:
        i -= 1
    FolderName = FolderName + "_" + str(i)
    if CheckpointPath is None:
        DEFolder = os.path.join("ModelCheckpoints", FolderName)
    else:
        DEFolder = CheckpointPath
    # Actually create the folders if they do not exist
    if not os.path.exists(FolderName):
        os.makedirs(FolderName)
    if not os.path.exists(DEFolder):
        os.makedirs(DEFolder)
    # Create Kinase Embedding class to get the list of available kinases and their embeddings
    KE = KinaseEmbedding(Family = Family, Group = Group, Pathway = Pathways, Kin2Vec=Kin2Vec, Enzymes = Enzymes)
    # Create the training dataset
    TrainDS = dataset()
    TrainDS.getdata(TrainData, KE, islabeled=True)
    TrainSeqEmbedded = TrainDS.Get_Embedded_Seqs(AminoAcidProperties, ProtVec) # Get the sequence embeddings
    #Normalize Training data
    if NormalizeDE:
        TrainSeqEmbeddedreshaped = TrainSeqEmbedded.reshape(TrainSeqEmbedded.shape[0], TrainSeqEmbedded.shape[1] * TrainSeqEmbedded.shape[2])
        SeqEmbedScaler = preprocessing.StandardScaler().fit(TrainSeqEmbeddedreshaped)
        TrainSeqEmbeddedreshaped = SeqEmbedScaler.transform(TrainSeqEmbeddedreshaped)
        TrainSeqEmbedded = TrainSeqEmbeddedreshaped.reshape(TrainSeqEmbedded.shape[0], TrainSeqEmbedded.shape[1], TrainSeqEmbedded.shape[2])
    if TestData != '':
        TestDS = dataset()
        #TestDS.getdata(TestData, KE, islabeled=True, MultiLabel=True)
        TestDS.getdata(TestData, KE, islabeled=True, MultiLabel=True)
        TestSeqEmbedded = TestDS.Get_Embedded_Seqs(AminoAcidProperties, ProtVec)
        if NormalizeDE:
            TestSeqEmbeddedreshaped = TestSeqEmbedded.reshape(TestSeqEmbedded.shape[0], TestSeqEmbedded.shape[1] * TestSeqEmbedded.shape[2])
            TestSeqEmbeddedreshaped = SeqEmbedScaler.transform(TestSeqEmbeddedreshaped)
            TestSeqEmbedded = TestSeqEmbeddedreshaped.reshape(TestSeqEmbedded.shape[0], TestSeqEmbedded.shape[1], TestSeqEmbedded.shape[2])
    if ValData != '':
        ValDS = dataset()
        ValDS.getdata(ValData, KE, islabeled=True, MultiLabel=True)
        ValSeqEmbedded = ValDS.Get_Embedded_Seqs(AminoAcidProperties, ProtVec)
        if NormalizeDE:
            ValSeqEmbeddedreshaped = ValSeqEmbedded.reshape(ValSeqEmbedded.shape[0], ValSeqEmbedded.shape[1] * ValSeqEmbedded.shape[2])
            ValSeqEmbeddedreshaped = SeqEmbedScaler.transform(ValSeqEmbeddedreshaped)
            ValSeqEmbedded = ValSeqEmbeddedreshaped.reshape(ValSeqEmbedded.shape[0], ValSeqEmbedded.shape[1], ValSeqEmbedded.shape[2])
    Candidatekinases, CandidatekinaseEmbeddings, Candidateindices, CandidateKE_to_Kinase, Candidate_UniProtIDs = KE.readKinases(TestKinaseCandidates)
    ValCandidatekinases, ValCandidatekinaseEmbeddings, ValCandidateindices, ValCandidateKE_to_Kinase, ValCandidate_UniProtIDs = KE.readKinases(ValKinaseCandidates)
    
    TrueClassIDX = FindTrueClassIndices(TrainDS.KinaseEmbeddings, TrainDS.UniqueKinaseEmbeddings)
    Test_TrueClassIDX = FindTrueClassIndices(TestDS.KinaseEmbeddings, CandidatekinaseEmbeddings, True)
    Val_TrueClassIDX = FindTrueClassIndices(ValDS.KinaseEmbeddings, ValCandidatekinaseEmbeddings, True)
        
        
    # Create Data Embedding model and train it over the training data
    EndToEndmodel = EndToEndModel(vocabnum = TrainSeqEmbedded.shape[-1], seqlens = TrainSeqEmbedded.shape[-2], Params=ModelParams, LogDir = FolderName, ckpt_dir = DEFolder, WriteEmbeddingVis=False, ClassEmbeddingsize=TrainDS.KinaseEmbeddings.shape[-1], seed=seed)
    if LoadModel:
        Train_accuracy, Train_Loss, epochs_completed = EndToEndmodel.loadmodel()
    else:
        #try:
        Train_accuracy, Train_Loss= EndToEndmodel.train(TrainSeqEmbedded, TrainDS.KinaseEmbeddings, TrainCandidateKinases=TrainDS.UniqueKinaseEmbeddings, epochcount=TrainingEpochs, 
                            ValDE= ValSeqEmbedded, ValCandidatekinaseEmbeddings=ValCandidatekinaseEmbeddings, ValCandidateKE_to_Kinase=ValCandidateKE_to_Kinase, ValKinaseUniProtIDs=ValDS.KinaseUniProtIDs, ValKinaseEmbeddings=ValDS.KinaseEmbeddings, ValCandidateUniProtIDs=ValCandidate_UniProtIDs,
                            TestDE=TestSeqEmbedded, TestCandidatekinaseEmbeddings=CandidatekinaseEmbeddings, CandidateKE_to_Kinase=CandidateKE_to_Kinase, TestCandidateUniProtIDs=Candidate_UniProtIDs, TestKinaseUniProtIDs=TestDS.KinaseUniProtIDs, TestKinaseEmbeddings=TestDS.KinaseEmbeddings, TrueClassIDX= TrueClassIDX, Test_TrueClassIDX= Test_TrueClassIDX, Val_TrueClassIDX = Val_TrueClassIDX)
    #except:
        #    with open(os.path.join(FolderName,'Error.txt'), 'w+') as f:
        #        print(str(EndToEndmodel._epochs_completed) + ' , ' + str(sys.exc_info()), file=f)
        #EndToEndmodel.savemodel()
    
    Train_accuracy, Train_Loss, epochs_completed = EndToEndmodel.loadmodel()
    Train_Evaluations = {"Accuracy":Train_accuracy, "Loss":Train_Loss}
    UniProtIDs, probabilities = EndToEndmodel.predict(TestSeqEmbedded, CandidatekinaseEmbeddings, CandidateKE_to_Kinase, WriteZSLWeights=False)
    UniProtIDs, probabilities = ensemble(UniProtIDs, probabilities, Candidate_UniProtIDs)
    
    Test_Evaluations = GetAccuracyMultiLabel(UniProtIDs, probabilities, TestDS.KinaseUniProtIDs, Test_TrueClassIDX)
    with open(os.path.join(FolderName,'Test_probabilities.txt'), 'w+') as f:
        np.savetxt(f, probabilities)
    
    ValUniProtIDs, Valprobabilities = EndToEndmodel.predict(ValSeqEmbedded, ValCandidatekinaseEmbeddings, ValCandidateKE_to_Kinase, WriteZSLWeights=False)
    ValUniProtIDs, Valprobabilities = ensemble(ValUniProtIDs, Valprobabilities, ValCandidate_UniProtIDs)
    
    Val_Evaluations = GetAccuracyMultiLabel(ValUniProtIDs, Valprobabilities, ValDS.KinaseUniProtIDs, Val_TrueClassIDX)
    
    with open(os.path.join(FolderName,'Val_probabilities.txt'), 'w+') as f:
        np.savetxt(f, Valprobabilities)
        
    #accuracy = accuracy_score(UniProtIDs, TestDS.KinaseUniProtIDs)
    print("Number of Train data: {} Number of Test data: {} Number of Val data: {}".format(len(TrainDS.Sub_IDs), len(TestDS.Sub_IDs), len(ValDS.Sub_IDs)))
    print("TrainLoss = ", Train_Loss, "ValLoss = ", Val_Evaluations["Loss"], "TestLoss = ", Test_Evaluations["Loss"])
    print("TrainAccuracy = ", Train_Evaluations["Accuracy"], "Valaccuracy = ", Val_Evaluations["Accuracy"], "TestAccuracy = ", Test_Evaluations["Accuracy"])
    
    return Train_Evaluations, Val_Evaluations, Test_Evaluations
    
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--BestParamethers', help='Use the best paramethers for the models', action='store_true')
    # The Zero-Shot learning model
    parser.add_argument('--Model', help='The default model for classification (ZSL, RNN, BiRNN)', default="SZL", type=str)
    parser.add_argument('--ModelEpochs', help='Number of Epochs for training the main model (ZSL, Logistic Regression or SVM)', default=100, type=int)
    # Data Embedding types
    parser.add_argument('--AminoAcidProperties', action='store_true', help='Use other properties for Amino Acids while embedding Options are True or False')
    parser.add_argument('--UseProtVec', action='store_true', help='Use ProtVec to convert the dataembedding sequence to a vector')
    # Data Embedding model paramethers
    parser.add_argument('--Embedding', help='Select the Embedding model the options are BiRNN and RNN do not write this argument for not using an embedding model', type=str, default="Binary")
    parser.add_argument('--RNNNodeType', help='The type of node to use in RNN model (CUDNNLSTM, LNlstm, lstm, GRU, None)', default="LNlstm", type=str)
    parser.add_argument('--EmbeddingDIM', help='The size of embedding layer at beginning', default=100, type=int)
    parser.add_argument('--DENumofLayers', help='The number of hidden layers in DataEmbedding model', default=1, type=int)
    parser.add_argument('--DEhiddenUnits', help='The number of hidden units in DataEmbedding model', default=100, type=int)
    parser.add_argument('--IncreaseEmbSize', default=0, type=int, help='Use a dense layer to increase the size of the embedding layer to given number' )
    parser.add_argument('--NumOfConvLayers', help='Number of filters for each Conv Layers to use in BiRNN before BiRNN model leave empty to not use Conv layers', nargs='*', type=int, default=[])
    parser.add_argument('--DoNotUseEmbeddingLayer', action='store_false', help='Do Not use Embedding Layer' )
    parser.add_argument('--UseAttention', action='store_true', help='Use Attention Mechanism in RNN and BiRNN models')
    parser.add_argument('--AttSize', help='Size of the attention layer',  default=5, type=int)
    parser.add_argument('--LoadDEModel', action='store_true', help='If included the data embedding model will be loaded from the appropriate folder and will not be trained' )
    # Kinase Embedding options
    parser.add_argument('--UseFamily', action='store_true', help='Use Kinase Family in Class Embedding')
    parser.add_argument('--UseGroup', action='store_true', help='Use Kinase Group in Class Embedding')
    parser.add_argument('--UsePathway', action='store_true', help='Use KEGG Pathways as class embedding')
    parser.add_argument('--UseEnzymes', action='store_true', help='Use Enzymes vectors as class embedding')
    parser.add_argument('--UseKin2Vec', action='store_true', help='Use Vectors generated from protein2vec for Kinase Sequences as class embedding')
    # Get data paths
    parser.add_argument('--TrainData', help='The path for train data', type=str, default='Data/Train_Phosphosite_new.txt')
    #parser.add_argument('--TestData', help='The path for test data', type=str, default='Data/Test_Phosphosite_MultiLabel.txt')
    # Test data Paths
    parser.add_argument('--TestData', help='The path for test data', type=str, default='Data/Test_Phosphosite_MultiLabel.txt')
    parser.add_argument('--TestKinaseCandidates', help='The path to the file which contains the list of test candidates', type=str, default='Data/Candidate_Kinases_new.txt')
    # Val data Paths
    parser.add_argument('--ValData', help='The path for validation data', type=str, default='Data/Val_Phosphosite_MultiLabel.txt')
    parser.add_argument('--ValKinaseCandidates', help='The path to the file which contains the list of test candidates', type=str, default='Data/Val_Candidate_Kinases.txt')
    
    parser.add_argument('--CustomLabelForResult', help='Add a custom label to the name of the method in output result',  default="", type=str)
    args=parser.parse_args()
    print(args)
    ModelParams = EndToEndModel.Params
    #ModelParams["regs"] = 0
    ModelParams["UseBatchNormalization1"] = True
    if args.BestParamethers:
        Run(Model = 'ZSL', TrainingEpochs = args.ModelEpochs,
            AminoAcidProperties = False, ProtVec = True, NormalizeDE = False,
            ModelParams= ModelParams, Family = True, Group = True, Pathways = False, Kin2Vec=True, Enzymes = True,
            LoadDEModel = args.LoadDEModel, CustomLabel=args.CustomLabelForResult,
            TrainData = args.TrainData, TestData = args.TestData, ValData=args.ValData, TestKinaseCandidates= args.TestKinaseCandidates, ValKinaseCandidates= args.ValKinaseCandidates)
    else:
        BiRNNfeatures = False
        RNNfeatures = False
        if args.Embedding == 'BiRNN':
            BiRNNfeatures = True
        elif args.Embedding == 'RNN':
            RNNfeatures = True
        elif args.Embedding == 'Binary':
            BiRNNfeatures = False
            RNNfeatures = False
        else:
            raise TypeError('Embedding should either be BiRNN or RNN or Binary (default)')
        
        Run(Model = args.Model, ModelEpochs = args.ModelEpochs,
            AminoAcidProperties = args.AminoAcidProperties, ProtVec = args.UseProtVec,
            BiRNNfeatures = BiRNNfeatures, Family = args.UseFamily, Group = args.UseGroup, Pathways = args.UsePathway, Kin2Vec=args.UseKin2Vec, Enzymes = args.UseEnzymes,
            UseEmbeddingLayer=args.DoNotUseEmbeddingLayer, ConvLayers = args.NumOfConvLayers, DataEmbeddingEpochs = args.DataEmbeddingEpochs, IncreaseEmbSize= args.IncreaseEmbSize, RNNNodeType= args.RNNNodeType, HiddenNodes = args.DEhiddenUnits, RNN_num_layers=args.DENumofLayers, EmbeddingDIM = args.EmbeddingDIM, BiRNNAttention= args.UseAttention, AttentionSize=args.AttSize,
            LoadDEModel = args.LoadDEModel, CustomLabel=args.CustomLabelForResult, ZSLseed=args.ZSLSeed,
            TrainData = args.TrainData, TestData = args.TestData, ValData=args.ValData, TestKinaseCandidates= args.TestKinaseCandidates)
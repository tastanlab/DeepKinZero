# -*- coding: utf-8 -*-
"""
This file contains the necessary classes and methods for reading files and preparing data for training and testing the models

@author: iman
"""
import csv
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import os

DataPath= "Data"
AllKinasePath = os.path.join(DataPath,"AllKinases.txt")
KinaseEmbeddingPath = os.path.join(DataPath,"KinaseFeatures.txt")
class dataset:
    """
    This class holds the data of substrates and their sequences, these can be pairs of labeled instances or unlabeled.
    It also generates different data embeddings.
    """
    AminoAcids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '_']
    
    Sequences= [] # The sequences of sites
    Sub_IDs= [] # The Uniprot IDs of Substrates
    Kinases= [] # The list of kinases it can be empty if this is a test set
    KinaseUniProtIDs= [] # The list of the kinase uniprot IDs
    SeqSize = 7 # The sequence size of the phosphosite is SeqSize * 2 + 1 this paramether indicates how many of amino acids we will pick in each side of the site
    
    Kin_One_Hot_Encoded= [] #The one hot encoded kinases will be used for training RNN
    
    Kinase_ToOneHot = {} #A dictionary for converting kinase class to one hot vector
    OneHot_ToKinase = {} #A dictionary for converting one hot one hot vector to kinase class
    
    UniqueKinases = [] #Unique kinases in this dataset the type is kinase classes
    UniqueKinaseEmbeddings = [] #Unique kinase embeddings
    UniqueKinaseCounts = [] #The count of each unique kinase in this dataset
    
    Num_of_UniqKin = 0 #Number of unique kinases in this dataset
    
    KinaseEmbeddings = [] #The Kinase embeddings of the kinases in this class
    
    TrigramToVec = {} #A dictionary holding the ProtVec vector for each possible trigram of amino acids
    
    protVecVectors = [] #A list of protvec vectors for each sequence in the dataset
    propertyVectors = [] #A list of vectors generated by amino acid properties for each sequence
    seqBinaryVectors = [] #A list of vectors generated by changing each amino acid into a one-hot binary vector
    def __init__(self):
        """
        The construct only reads trigram vectors to initialize the dictionaries
        """
        
        self.Sequences= [] # The sequences of sites
        self.Sub_IDs= [] # The Uniprot IDs of Substrates
        self.Residues = []
        self.Kinases= [] # The list of kinases it can be empty if this is a test set
        self.SeqSize = 7 # The sequence size of the phosphosite is SeqSize * 2 + 1 this paramether indicates how many of amino acids we will pick in each side of the site
    
        self.Kin_One_Hot_Encoded= [] #The one hot encoded kinases will be used for training RNN
    
        self.Kinase_ToOneHot = {} #A dictionary for converting kinase class to one hot vector
        self.OneHot_ToKinase = {} #A dictionary for converting one hot one hot vector to kinase class
    
        self.UniqueKinases = [] #Unique kinases in this dataset the type is kinase classes
        self.UniqueKinaseEmbeddings = [] #Unique kinase embeddings
        self.UniqueKinaseCounts = [] #The count of each unique kinase in this dataset
    
        self.Num_of_UniqKin = 0 #Number of unique kinases in this dataset
    
        self.KinaseEmbeddings = [] #The Kinase embeddings of the kinases in this class
    
        self.TrigramToVec = {} #A dictionary holding the ProtVec vector for each possible trigram of amino acids
    
        self.protVecVectors = [] #A list of protvec vectors for each sequence in the dataset
        self.seqBinaryVectors = []
        self.propertyVectors = []
        
        self.readTrigramVectos()
        
    def getdata(self, datapath, AllKinases, islabeled=False, MultiLabel=False):
        """
        This methods reads the given data and fills the paramethers
        
        Args:
            datapath (string): the path of the dataset file, the file should be a csv file and its columns should be substrate_ID, position, sequence, Kinase_UniprotID(optional)
            islabeled (bool): boolean value shows if the dataset have labels or not (the labels should be on last column)
        """
        self.MultiLabel = MultiLabel
        self.protVecVectors = []
        self.Kinases = []
        self.KinaseUniProtIDs = []
        with open(datapath) as csvfile:
            Sub_DS = csv.reader(csvfile, delimiter='\t')
            #next(Sub_DS, None) # Skip the header
            for row in Sub_DS:
                if islabeled and not MultiLabel:
                    # Check if the kinase is in All_kinases file, if it is not, skip the line
                    if row[3] not in AllKinases.UniProtID_to_Kinase:
                        print(row[3], " Not Found in Allkinases dataset skipping row:", row)
                        continue
                self.Sub_IDs.append(row[0])
                self.Residues.append(row[1])
                Sequence = list(row[2].upper())
                self.Sequences.append(Sequence)
                if islabeled:
                    if not MultiLabel:
                        kinase = AllKinases.UniProtID_to_Kinase[row[3]]
                        self.Kinases.append(kinase)
                        self.KinaseUniProtIDs.append(row[3])
                        if kinase not in self.UniqueKinases:
                            self.UniqueKinases.append(kinase)
                    else:
                        curkinases = []
                        UniProtIDs = row[3].split(',')
                        for ID in UniProtIDs:
                            kinase = AllKinases.UniProtID_to_Kinase[ID]
                            curkinases.append(kinase)
                            if kinase not in self.UniqueKinases:
                                self.UniqueKinases.append(kinase)
                        self.Kinases.append(curkinases)
                        self.KinaseUniProtIDs.append(UniProtIDs)
                vecmat = self.getProtVecVectors(Sequence)
                self.protVecVectors.append(self.Pad_Vectors(Sequence, vecmat))
            if islabeled:
                self.createOneHotClassEmbeddings()
            self.protVecVectors = np.array(self.protVecVectors)
            self.setonehotencodedSeq(self.Sequences)
            AA = AminoAcids()
            self.propertyVectors, newlength = AA.getonehotallalphabet(self)
            
    def createOneHotClassEmbeddings(self):
        """
        Create One hot class embedding for each kinase and store them in the paramethers in the class
        """
        #self.UniqueKinases, self.UniqueKinaseCounts = np.unique(self.Kinases,return_counts=True)
        self.Num_of_UniqKin = len(self.UniqueKinases)
        ID = 0
        for UniqKin in self.UniqueKinases:
            OneHot = np.zeros([self.Num_of_UniqKin])
            OneHot[ID] = 1
            ID += 1
            self.Kinase_ToOneHot[UniqKin] = OneHot
            self.UniqueKinaseEmbeddings.append(UniqKin.EmbeddedVector)
        for kin in self.Kinases:
            if not self.MultiLabel:
                self.Kin_One_Hot_Encoded.append(self.Kinase_ToOneHot[kin])
                self.KinaseEmbeddings.append(kin.EmbeddedVector)
            else:
                onehots = []
                kinaseembeddings = []
                for k in kin:
                    onehots.append(self.Kinase_ToOneHot[k])
                    kinaseembeddings.append(k.EmbeddedVector)
                self.Kin_One_Hot_Encoded.append(onehots)
                self.KinaseEmbeddings.append(kinaseembeddings)
        self.KinaseEmbeddings = np.array(self.KinaseEmbeddings)
        self.Kin_One_Hot_Encoded = np.array(self.Kin_One_Hot_Encoded)
        self.UniqueKinaseEmbeddings = np.array(self.UniqueKinaseEmbeddings)
    
    def readTrigramVectos(self):
        """
        Read the file which contains the protvec vectors for all the possible trigrams and store it in a dictionary
        """
        with open(os.path.join(DataPath, 'Allcomb2Vec.txt'), 'r') as f:
            for line in f:
                line = line.rstrip()
                splits = line.split('\t')
                Trigram = splits[0]
                splitted = splits[1].split(" , ")
                splitted = list(map(float, splitted))
                self.TrigramToVec[Trigram] = splitted
            
    def getProtVecVectors(self, Sequence):
        """
        Given a sequence return the protvec vectors for each trigram
        
        Args:
            Sequence (str):The sequence of amino acids to generate trigrams for
        
        Return:
            vecmat (list): list of protvec vectors for each trigram in the sequence, the size of each protvec vector is 100
        """
        vecmat = []
        n_grams = self.gettrigrams(Sequence)
        for grams in n_grams:
                if "_" not in grams:
                    vecmat.append(self.TrigramToVec[''.join(grams)])
        return vecmat
    
    def gettrigrams(self, Sequence):
        """
        Return all trigrams of a sequence
        
        Args:
            Sequence (str): The sequence to generate the trigrams for
        
        Return:
            out (list): list of all trigrams
        """
        out = []
        for i in range(len(Sequence)-2):
            out.append(Sequence[i:(i+3)])
        return out
    
    def Pad_Vectors(self, Sequence, vec):
        """
        Method to pad ProtVec Vectors based on '_' in the sequence 
        so if there is any '_' in any of the trigrams it will be restored with a array of zeros of length 100
        
        Args:
            Sequence (str): The input sequence
            vec (list): The input ProtVec vector
            
        Return:
            outvec (str): The padded ProtVec vector
        """
        outvec = []
        j = 0
        for i in range(len(Sequence) - 2):
            if '_' in Sequence[i:i+3]:
                outvec.append(np.zeros(100).tolist())
            else:
                outvec.append(vec[j])
                j = j + 1
        return outvec
    
    def setonehotencodedSeq(self, Sequences):
        """
        create one-hot binary encoded sequence for each sequence in the Sequences array and set it as binarryseqarray
        
        Args:
            Sequences (list): list of sequences of amino acids
        """
        self.integer_encoded_seqs = []
        for seq in Sequences:
            intseq = []
            for c in seq:
                intseq.append(self.AminoAcids.index(c))
            self.integer_encoded_seqs.append(intseq)
        self.integer_encoded_seqs = np.array(self.integer_encoded_seqs)
        self.seqBinaryVectors = self.getonehotencodedSeq(self.AminoAcids, Sequences)
    
    def getonehotencodedSeq(self, chars, Sequences):
        """
        Create one-hot binary sequence for each sequence in sequences and returns it
        
        Args:
            Sequences (list): list of sequences of amino acids
        """
        alphabet = sorted(set(chars))
        onehottedseq = []
        for seq in Sequences:
            onehottedseq.append(self.string_vectorizer(seq, alphabet))
        onehotseqarray = []
        for i in range(0,len(onehottedseq)):
            onehotseqarray.append(np.concatenate(onehottedseq[i],0))
        onehotseqarray = np.array(onehotseqarray)
        return onehotseqarray
    
    def string_vectorizer(self,strng, alphabet):
        vector = [[0 if char != letter else 1 for char in alphabet] 
                      for letter in strng]
        return vector
    
    def Get_Embedded_Seqs(self, AminoAcidProperties, ProtVec):
        """
        Get the embedded sequences as a list of vectors
        
        Args:
            AminoAcidProperties (binary): return sequences as represented by one-hot binary vectors for each amino acid
            ProtVec (binary): return sequences as represented by protvec vectors
        
        Return:
            a list of vectors generated for each sequence in the dataset, the length of vectors is different based on the selected method
            Protvec will generate vectors of length 100, AminoAcidProperties will generate vectors of length  and binary will generate vectors of length 315
        """
        if ProtVec:
            return self.protVecVectors
        elif AminoAcidProperties:
            return self.propertyVectors
        else:
            return self.seqBinaryVectors.reshape([-1, 15, 21])
    @staticmethod
    def Get_SeqSize(AminoAcidProperties, ProtVec):
        """
        According to given flags AminoAcidProperties and ProtVec, calculates the size of output
        The output is a tuple of (Seq_size, EmbeddingSize)
        
        Args:
            AminoAcidProperties (binary): return sequences as represented by one-hot binary vectors for each amino acid
            ProtVec (binary): return sequences as represented by protvec vectors
        
        Return:
            a tuple of sequence (SequenceSize, EmbeddingSize)
        """
        if ProtVec:
            return (13, 100)
        elif AminoAcidProperties:
            return (15, 16)
        else:
            return (15, 21)
class kinase:
    """
    Class for holding kinase information
    """
    def __init__(self, _Protein_Name,_EntrezID, _UniprotID, _EmbeddedVector = None):
        self.Protein_Name= _Protein_Name
        self.EntrezID= _EntrezID
        self.UniprotID= _UniprotID
        if _EmbeddedVector != None:
            self.EmbeddedVector = _EmbeddedVector
            
    def __lt__(self, otherKinase):
        """
        Making the kinase sortable by its name
        
        Args:
            otherKinase (kinase): Other kinase to compare to
        Return:
            True if the name of this kinase is smaller than the otherKinase
        """
        return (self.Protein_Name < otherKinase.Protein_Name)
    
    def __eq__(self, otherKinase):
        """
        Making the kinases comparable
        
        Args:
            otherKinase (kinase): Other kinase to compare to
        Return:
            True if kinases are equal and false otherwise
        """
        # it is enough to compare the uniprotIDs since it is unique for every protein
        return (self.UniprotID == otherKinase.UniprotID)
        #return (self.Protein_Name == otherKinase.Protein_Name)
        
    def __hash__(self):
        """
        Making the kinases hashable
        
        Return:
            hash of UniprotID
        """
        return hash(self.UniprotID)
        #return hash(self.Protein_Name)
class KinaseEmbedding:
    """
    This class holds the list of all kinases and create the kinase embeddings for each one
    """
    allkinases = [] # the list of all kinases
    UniProtID_to_Kinase = {} # a dictionary of uniprotID to kinase
    AllKinaseEmbeddings = [] # A list of all kinases embedded vectors
    KE_to_Kinase = {} # a dictionary of kinase Embedding vector to kinase class
    Embedding_size = 0 # The size of Kinase embeddings
    def __init__(self, Family = True, Group = True, Pathway = True, Kin2Vec = True, InterProDomains = True, Enzymes = True, SubCellLoc = True, GO_C_vec = True, GO_F_vec = True, GO_P_vec = True):
        """
        Get the paramethers for class embedding and initilize some variables to use later then run readKinaseEmbedding and ReadKinases methods to read all the kinases and create the embeddings for them
        
        Args:
            Family (bool): Use Kinase family data in Class Embeddings
            Group (bool): Use Kinase group (superfamily) data in Class Embeddings
            Pathway (bool): Use KEGG pathways data in Class Embeddings
            Kin2Vec (bool): Use Vectors generated from ProtVec for Kinase Sequences in Class Embeddings
            InterProDomains (bool): Use InterPro domains in Class Embedding
            Enzymes (bool): Use Enzymes hierarchy in Class Embedding
            SubCellLoc (bool): Use Sub cellular localisation in Class Embedding
            GO_C_vec: Use GO cellular component analysis in Class Embedding
            GO_F_vec: Use GO functional in Class Embedding
            GO_P_vec: Use GO pathways in Class Embedding
        """
        self.Family= Family; self.Group=Group; self.Pathway= Pathway; self.Kin2Vec = Kin2Vec; self.InterProDomains= InterProDomains; self.Enzymes= Enzymes; self.GO_C_vec= GO_C_vec; self.GO_F_vec= GO_F_vec; self.GO_P_vec= GO_P_vec
        self.readKinaseEmbedding()
        self.ReadKinases()
        self.Embedding_size = len(self.AllKinaseEmbeddings[0])
        
    def ReadKinases(self):
        """
        This method reads all kinases from the AllKinasePath and creates the necessary dictionaries and arrays
        """
        with open(AllKinasePath, 'r') as AllKinasefilecsv:
            AllKinasefile= csv.reader(AllKinasefilecsv, delimiter='\t')
            for row in AllKinasefile:
                newkinase = kinase(row[0], row[1], row[2])
                self.allkinases.append(newkinase)
                self.UniProtID_to_Kinase[row[2]] = newkinase
            self.AllKinaseEmbeddings, _ = self.createclassembedding(self.allkinases)
    def makeonehotencoded(self,classes):
        """
        Gets an array and returns one hot encoded version of it
        
        Args:
            classes (string array): The input strings that will be one hot encoded
        """
        # Convert the array to numpy array
        values = np.array(classes)
        # Define a encoder
        label_encoder = LabelEncoder()
        # Convert the labels to integers
        integer_encoded = label_encoder.fit_transform(values)
        # binary encode
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        
        return onehot_encoded, label_encoder
    def readKinaseEmbedding(self):
        """
        Read the kinase embeddings file to get the features of kinases
        """
        with open(KinaseEmbeddingPath) as csvfile:
            KinEmb = csv.reader(csvfile, delimiter='\t')
            #next(KinEmb, None) # Pass the header
            families = []
            groups = []
            KinUniProtIDs = []
            EnzymesVecs = []
            Domains = []
            Kin2Vecs = []
            Pathways = []
            # Read the kinase embedding file
            for row in KinEmb:
                KinUniProtID = row[0]
                KinUniProtIDs.append(KinUniProtID)
                Family = row[1]
                families.append(Family)
                Group = row[2]
                groups.append(Group)
                EnzymesVec = row[3]
                EnzymesVec = list(EnzymesVec)
                EnzymesVec = list(map(int,EnzymesVec))
                EnzymesVecs.append(EnzymesVec)
                Domain = row[4]
                Domains.append(Domain)
                Kin2Vec = row[5]
                Kin2Vec = Kin2Vec.split(", ")
                Kin2Vec = list(map(float, Kin2Vec))
                Kin2Vecs.append(Kin2Vec)
                Pathway = row[6]
                Pathway = Pathway.split(", ")
                Pathway = list(map(int,map(float, Pathway)))
                Pathways.append(Pathway)
            # Create dictionary for each of the embeddings
            self.Kinonehot_encoded, self.KinaseEncoder = self.makeonehotencoded(KinUniProtIDs) # Convert UniProtIDs to one-hot encoded
            self.Familyonehot_encoded, self.familyEncoder = self.makeonehotencoded(families) # Convert families to one-hot encoded vectors
            self.Grouponehot_encoded, self.groupEncoder = self.makeonehotencoded(groups) # Convert groups to one-hot encoded vectors
            self.UniProtID_to_OneHotVec = dict(zip(KinUniProtIDs,self.Kinonehot_encoded))
            self.UniProtID_to_FamilyVec= dict(zip(KinUniProtIDs,self.Familyonehot_encoded))
            self.UniProtID_to_GroupVec= dict(zip(KinUniProtIDs,self.Grouponehot_encoded))
            self.UniProtID_to_Kin2Vec= dict(zip(KinUniProtIDs, Kin2Vecs))
            self.UniProtID_to_EnzymesVec = dict(zip(KinUniProtIDs, EnzymesVecs))
            self.UniProtID_to_Pathway= dict(zip(KinUniProtIDs, Pathways))
            
    def getEmbedding(self, UniprotID):
        """
        Get embedding vector for a single kinase
        
        Args:
            UniprotID (int): UniProt ID of the kinase to generate its embedding
        
        Return:
            The embedded vector of the input kinase
        """
        ClassEmbedding = self.UniProtID_to_OneHotVec[UniprotID]
        if self.Group:
            ClassEmbedding = np.append(ClassEmbedding,self.UniProtID_to_GroupVec[UniprotID])
        if self.Family:
            ClassEmbedding = np.append(ClassEmbedding,self.UniProtID_to_FamilyVec[UniprotID])
        if self.Pathway:
            ClassEmbedding = np.append(ClassEmbedding,self.UniProtID_to_Pathway[UniprotID])
        if self.Kin2Vec:
            ClassEmbedding = np.append(ClassEmbedding,self.UniProtID_to_Kin2Vec[UniprotID])
        if self.Enzymes:
            ClassEmbedding = np.append(ClassEmbedding,self.UniProtID_to_EnzymesVec[UniprotID])
        return ClassEmbedding
    
    def createclassembedding(self, kinases):
        """
        Get class embedding for each kinase in the given list and store them in kinase embedded vector and also return a list containing these class embeddings
        
        Args:
            kinases (list of kinase): The list of kinases to produce class embedding for
        
        Return:
            KinaseEmbeddings list which is the sorted class embeddings for each of the input kinases
            UniqueClassEmbedding list of unique class embeddings
        """
        KinaseEmbeddings = []
        for kin in kinases:
            kin.EmbeddedVector = self.getEmbedding(kin.UniprotID)
            KinaseEmbeddings.append(kin.EmbeddedVector)
            self.KE_to_Kinase[tuple(kin.EmbeddedVector)] = kin
            
        UniqueClassEmbedding = np.vstack({tuple(row) for row in KinaseEmbeddings})
        return KinaseEmbeddings, UniqueClassEmbedding
    
    def readKinases(self, path):
        """
        Read kinases from a given path
        
        Args:
            path(str): path of the file to read the kinases from. The file should contain a single column which contains the uniprotIDs of kinases
        
        Returns:
            kinases(list of kinase): a list of kinases in the file
            kinaseEmbeddings(list of list): a list of kinase embeddings
            indices(list of ints): indices of kinases in the allkinase file
        """
        indices = []
        kinases = []
        kinaseEmbeddings = []
        KE_to_Kinase = {}
        UniProtIDs = []
        with open(path, 'r') as kinasefile:
            for UniProtID in kinasefile:
                UniProtID = UniProtID.strip()
                UniProtIDs.append(UniProtID)
                kinase = self.UniProtID_to_Kinase[UniProtID]
                kinases.append(kinase)
                kinaseEmbeddings.append(kinase.EmbeddedVector)
                indices.append(self.allkinases.index(kinase))
                KE_to_Kinase[tuple(kinase.EmbeddedVector)] = UniProtID
        return np.array(kinases), np.array(kinaseEmbeddings), np.array(indices), KE_to_Kinase, UniProtIDs
    def get_UniProtIDs_from_KE(self, KinaseEmbeddings):
        """
        Get uniprotIDs of given kinase embeddings
        
        Args:
            KinaseEmbeddings(list of list): the list of kinase embeddings to return their uniprotIDs
        """
        UniProtIDs =[]
        for KE in KinaseEmbeddings:
            UniProtIDs.append(self.KE_to_Kinase[tuple(KE)].UniprotID)
        return np.array(UniProtIDs)

class AminoAcids:
    Charge = {"A":"U", "C":"U", "D": "N", "E": "N", "F":"U", "G":"U", "H" : "P", "I":"U", "K" : "P", "L":"U", "M":"U", "N":"U", "P":"U", "Q":"U", "R" : "P", "S":"U", "T":"U", "V":"U", "W":"U", "Y":"U"} # "*": "U"
    Polarity = {"A":"N", "C":"P", "D": "P", "E": "P", "F":"N", "G":"N", "H" : "P", "I" : "N", "K" : "P", "L":"N", "M":"N", "N":"P", "P" : "N", "Q" : "P", "R" : "P", "S":"P", "T":"P", "V":"N", "W":"N", "Y":"P"}# , "*": "P"
    Aromaticity = {"A":"N", "C":"N", "D": "N", "E": "N", "F":"R", "G":"N", "H" : "R", "I" : "L", "K" : "N", "L":"L", "M":"N", "N":"N", "P" : "N", "Q" : "N", "R" : "N", "S":"N", "T":"N", "V":"L", "W":"R", "Y":"R"}# , "*": "N"
    Size = {"A":"S", "C":"L", "D": "M", "E": "L", "F":"L", "G":"S", "H" : "L", "I" : "L", "K" : "L", "L":"L", "M":"L", "N":"M", "P" : "S", "Q" : "L", "R" : "L", "S" : "S", "T" : "M", "V":"L", "W":"L", "Y":"L"}# , "*": "P"
    #strong donor(S) (A, D, E, P), weak donor(W) (I, L, V), neutral(N) (C, G, H, S, W), weak acceptor(A) (F, M, Q, T, Y), strong acceptor(C) (K, N, R)
    Electricity = {"A":"S", "C":"N", "D": "S", "E": "S", "F":"A", "G":"N", "H" : "N", "I" : "W", "K" : "C", "L":"W", "M":"A", "N":"C", "P" : "S", "Q" : "A", "R" : "C", "S":"N", "T":"A", "V":"W", "W":"N", "Y":"A"}# , "*": "N"
    def changealphabet(self, word, alphabet):
        newalphabet = []
        newword = ""
        for c in word:
            if c != '_':
                newword = newword + alphabet[c]
            else:
                newword = newword + c
        return list(newword)
    def changealldataset(self, dataset, alphabet):
        """
        Given a dataset change all the amino acids to their equivalant in given alphabet dictionary
        """
        newdataset = []
        for seq in dataset:
            newdataset.append(self.changealphabet(seq,alphabet))
        newalphabets = np.unique(list(alphabet.values()))
        return newdataset, newalphabets
    def getonehotallalphabet(self, dataset):
        """
        This method gets a dataset and makes each sequence into an embedded vector with amino acid properties
        
        Args:
            dataset (dataset): the dataset to generate the vectors for
        
        Return:
            newdatset (list): the list of sequences embedded with amino acid properties
            newlength (int): the length of the new sequence 
        """
        newlength = 0
        newdatset = dataset.seqBinaryVectors
        seqlength = 2 * dataset.SeqSize + 1
        newdatset = newdatset.reshape(len(newdatset),seqlength, len(dataset.AminoAcids))
        newlength += len(dataset.AminoAcids)
        Changed, newalphabets = self.changealldataset(dataset.Sequences, self.Charge)
        ChangedOneHotted = dataset.getonehotencodedSeq(newalphabets,Changed)
        ChangedOneHottedreshaped = ChangedOneHotted.reshape(len(ChangedOneHotted), seqlength, len(newalphabets))
        newlength += len(newalphabets)
        newdatset = np.append(newdatset, ChangedOneHottedreshaped, axis=2)
        Changed, newalphabets = self.changealldataset(dataset.Sequences, self.Polarity)
        ChangedOneHotted = dataset.getonehotencodedSeq(newalphabets,Changed)
        ChangedOneHottedreshaped = ChangedOneHotted.reshape(len(ChangedOneHotted), seqlength, len(newalphabets))
        newlength += len(newalphabets)
        newdatset = np.append(newdatset, ChangedOneHottedreshaped, axis=2)
        Changed, newalphabets = self.changealldataset(dataset.Sequences, self.Aromaticity)
        ChangedOneHotted = dataset.getonehotencodedSeq(newalphabets,Changed)
        ChangedOneHottedreshaped = ChangedOneHotted.reshape(len(ChangedOneHotted), seqlength, len(newalphabets))
        newlength += len(newalphabets)
        newdatset = np.append(newdatset, ChangedOneHottedreshaped, axis=2)
        Changed, newalphabets = self.changealldataset(dataset.Sequences, self.Size)
        ChangedOneHotted = dataset.getonehotencodedSeq(newalphabets,Changed)
        ChangedOneHottedreshaped = ChangedOneHotted.reshape(len(ChangedOneHotted), seqlength, len(newalphabets))
        newlength += len(newalphabets)
        newdatset = np.append(newdatset, ChangedOneHottedreshaped, axis=2)
        Changed, newalphabets = self.changealldataset(dataset.Sequences, self.Electricity)
        ChangedOneHotted = dataset.getonehotencodedSeq(newalphabets,Changed)
        ChangedOneHottedreshaped = ChangedOneHotted.reshape(len(ChangedOneHotted), seqlength, len(newalphabets))
        newlength += len(newalphabets)
        newdatset = np.append(newdatset, ChangedOneHottedreshaped, axis=2)
        return newdatset, newlength
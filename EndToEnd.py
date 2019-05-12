# -*- coding: utf-8 -*-

""" End to End model """
import tensorflow as tf
import numpy as np
import os, time
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
import random

#from Visualization import EmbeddingVisualization

from sklearn.metrics import accuracy_score, classification_report

from sklearn.preprocessing import MultiLabelBinarizer
#from BNlstm import BN_LSTMCell
#from tqdm import tqdm

def ensemble(UniProtIDs, probabilities, CandidatekinaseUniProtIDs):
    sum_probs = np.sum(np.array(probabilities), axis=0) / len(probabilities)
    outclassindx = np.argmax(sum_probs, axis=1)
    CandidatekinaseUniProtIDs = np.array(CandidatekinaseUniProtIDs)
    outUniprotIDs = CandidatekinaseUniProtIDs[outclassindx]
    return outUniprotIDs, sum_probs
def get_top_n(n, matrix):
    """Gets probability a number n and a matrix, 
    returns a new matrix with largest n numbers in each row of the original matrix."""
    return (-matrix).argsort()[:,0:n]

def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 

def GetAccuracyMultiLabel(Y_Pred, Probabilities, Y_True, TrueClassIndices, eps=1e-15):
        """ Returns Accuracy when multi-label are provided for each instance. It will be counted true if predicted y is among the true labels
        Args:
            Y_Pred (int array): the predicted labels
            Probabilities (float [][] array): the probabilities predicted for each class for each instance
            Y_True (int[] array): the true labels, for each instance it should be a list
        """
        count_true = 0
        count_true_3 = 0
        count_true_5 = 0
        count_true_10 = 0
        logloss = 0.0
        top_10_classes = get_top_n(10, Probabilities)
        for i in range(len(Y_Pred)):
            for idx in TrueClassIndices[i]:
                p = np.clip(Probabilities[i][idx], eps, 1 - eps)
                logloss -= np.log(p)
            if Y_Pred[i] in Y_True[i]:
                count_true += 1
            if len(intersection(top_10_classes[i], TrueClassIndices[i])) > 0:
                count_true_10 += 1
            if len(intersection(top_10_classes[i][:5], TrueClassIndices[i])) > 0:
                count_true_5+=1
            if len(intersection(top_10_classes[i][:3], TrueClassIndices[i])) > 0:
                count_true_3+=1
        
        Evaluations = {"Accuracy":(float(count_true) / len(Y_Pred)), "Loss":(float(logloss) / len(Y_Pred)), 
                       "Top3Acc":(float(count_true_3) / len(Y_Pred)) ,"Top5Acc":(float(count_true_5) / len(Y_Pred)) ,
                       "Top10Acc":(float(count_true_10) / len(Y_Pred)), "Probabilities":Probabilities, 
                       "Ypred":Y_Pred, "Ytrue":Y_True}
        return Evaluations
def LinearActivation(_x):
    return _x
def prelu(_x, scope=None):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)
class EndToEndModel:
    """
    rnn_unit_type = 'lstm' #The cell type that should be used for RNN it can be gru, lstm, LNlstm, CUDNNLSTM and None
    num_layers = 1 #The number of layers for BiRNN
    num_hidden_units = 100 #The number of hidden units to use for each layer of BiRNN
    vocabnum = 21 #The dimension of each 
    classnumber = 2 #The number of output Classes
    dropoutval = 0.5 #Dropout value
    learningrate = 0.01 #learning rate
    seq_lens = 15 #Sequence length of each input
    
    num_examples = 0 #Number of Examples
    training_epochs = 200 #Number of training epochs
    batch_size = 64 #batch size for each mini_batch
    display_step = 10 #The interval to test the model and show output
    
    EMBEDDING_DIM = 100 #the dimension of embedding layer
    ATTENTION_SIZE = 5 #Attension size
    
    DELTA = 0.5
    
    useEmbeddingLayer = True #Should the model use an embedding layer in beginning?
    
    IncreaseEmbSize= 0 #Should the model use a Dense layer on output before returning the embeddings? If it is bigger than 0 it will be the dimension of that output
    
    """
    LogDir = "" #the folder to write outputs
    ckpt_dir = "" #Where to write checkpoints
    
    loss = []
    
    
    Params = {"rnn_unit_type": "LNlstm", "num_layers": 1, "num_hidden_units": 500, "dropoutval": 0.5, "learningrate": 0.003, "useAtt": False, "useEmbeddingLayer": False, "useEmbeddingLayer": False, "num_of_Convs": [], "UseBatchNormalization1": True, "UseBatchNormalization2": True, "EMBEDDING_DIM": 500, "ATTENTION_SIZE": 5, "IncreaseEmbSize": 0, "Bidirectional":True, "Dropout1": True, "Dropout2": True, "Dropout3": False, "regs": 0.001, "batch_size": 64, "ClippingGradients": 9.0, "activation1": "tanh", "LRDecay":True, "seed":None, "NumofModels": 3} #a dictionary indicating the parameters provided for the model
    
    seed = None
    
    ClassEmbeddingsize = 0
    
    def __init__(self, vocabnum, ClassEmbeddingsize, Params=None, seqlens = 15, sess = None, LogDir = None, ckpt_dir = None, seed = None, WriteEmbeddingVis=False):
        if Params != None:
            self.Params = Params
        #self.seed = self.Params["seed"]
        tf.reset_default_graph()
        #os.environ['PYTHONHASHSEED'] = '0'
        #tf.set_random_seed(self.seed)
        #np.random.seed(self.seed)
        #random.seed(self.seed)
        #tf.set_random_seed(seed)
        if Params["seed"] is not None:
            self.seed = [Params["seed"] + (i * 3) for i in range(self.Params["NumofModels"])]
        else:
            self.seed = [None for i in range(self.Params["NumofModels"])]
        self.vocabnum = vocabnum
        self.seq_lens = seqlens
        
        self.ModelSaved = False
        #self.useResidualWrapper = useResidualWrapper
        
        self.ClassEmbeddingsize = ClassEmbeddingsize
        
        self.WriteEmbeddingVis = WriteEmbeddingVis
        if LogDir is not None:
            self.ckpt_dir = ckpt_dir
            self.LogDir = LogDir
        self.Models = []
        self.Graphs = []
        
        self.batch_ph = []
        self.seq_len_ph = []
        self.keep_prob_ph = []
        self.is_training = []
        
        self.CE = []
        self.CKE = []
        self.TCI = []
        self.alphas = []
        
        self.saver = []
        
        self.train_writer = {'Ensemble': tf.summary.FileWriter(os.path.join(self.LogDir,'Ensemble','logdir','Ensemble', 'train'))}
        self.test_writer = {'Ensemble': tf.summary.FileWriter(os.path.join(self.LogDir,'Ensemble','logdir','Ensemble', 'test'))}
        self.val_writer = {'Ensemble': tf.summary.FileWriter(os.path.join(self.LogDir,'Ensemble','logdir','Ensemble', 'val'))}
        
        for i in range(self.Params["NumofModels"]):
            Graph = tf.Graph()
            self.Graphs.append(Graph)
            self.Models.append(self.create_graph(i))
        if sess is not None:
            self.sess = sess
        else:
            self.sess = []
            for i in range(self.Params["NumofModels"]):
                with self.Graphs[i].as_default():
                    #config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
                    config = tf.ConfigProto()
                    config.gpu_options.allow_growth = True
                    self.sess.append(tf.Session(graph=tf.get_default_graph(), config = config))
                    self.saver.append(tf.train.Saver())
        self.initilize()
        #self.Params = {"rnn_unit_type": rnn_unit_type, "num_layers": num_layers, "num_hidden_units": num_hidden_units, "vocabnum": vocabnum, "classnumber": classnumber, "dropoutval": dropoutval, "learningrate": learningrate, "seqlens": seqlens, "useAtt": useAtt,  "useEmbeddingLayer": useEmbeddingLayer, "useConvLayer": useConvLayer, "num_of_Convs":  num_of_Convs, "useResidualWrapper": useResidualWrapper, "UseBatchNormalization": UseBatchNormalization, "EMBEDDING_DIM": EMBEDDING_DIM, "ATTENTION_SIZE": ATTENTION_SIZE, "IncreaseEmbSize": IncreaseEmbSize, "Seed": seed}
        
    def rnn_cell(self, GraphID, fworbw, L=0):
        with tf.variable_scope('RNN_' + str(GraphID)):
            # Get the cell type
            if self.Params["rnn_unit_type"] == 'rnn':
                rnn_cell_type = tf.nn.rnn_cell.BasicRNNCell
            elif self.Params["rnn_unit_type"] == 'gru':
                rnn_cell_type = tf.nn.rnn_cell.GRUCell
            elif self.Params["rnn_unit_type"] == 'lstm':
                rnn_cell_type = tf.nn.rnn_cell.LSTMCell
            elif self.Params["rnn_unit_type"] == 'LNlstm':
                rnn_cell_type = tf.contrib.rnn.LayerNormBasicLSTMCell
            elif self.Params["rnn_unit_type"] == 'CUDNNLSTM':
                rnn_cell_type = tf.contrib.cudnn_rnn.CudnnLSTM
            else:
                raise Exception("Choose a valid RNN unit type.")
            
            #if self.rnn_unit_type == 'LNlstm':
            #Create a layer
            #For some strange reason when activation=tf.nn.relu is set the results become undeterminstic
            if L == self.Params["num_layers"] - 1:
                single_cell = rnn_cell_type(self.Params["num_hidden_units"], activation=LinearActivation)#, scope='RNN_Cell_'+ fworbw + '_' + str(L) + str(GraphID))
            else:
                if self.Params["activation1"] == "None":
                    single_cell = rnn_cell_type(self.Params["num_hidden_units"], activation=LinearActivation)#, scope='RNN_Cell_'+ fworbw + '_' + str(L) + str(GraphID))
                else:
                    single_cell = rnn_cell_type(self.Params["num_hidden_units"])#, scope='RNN_Cell_'+ fworbw + '_' + str(L) + str(GraphID))
            
            # Dropout
            #single_cell = tf.nn.rnn_cell.DropoutWrapper(single_cell, input_keep_prob = self.keep_prob_ph, output_keep_prob = self.keep_prob_ph, seed = self.seed)
            
            #if self.useResidualWrapper:
            #    single_cell = tf.contrib.rnn.ResidualWrapper(single_cell)
            
        
            # Each state as one cell
            #stacked_cell = tf.nn.rnn_cell.MultiRNNCell(
            #    [single_cell] * self.num_layers)
            #stacked_cell = [single_cell] * self.num_layers
    
        return single_cell
    
    def _add_conv_layers(self,inputs):
        """Adds convolution layers."""
        convolved = inputs
        for i in range(len(self.Params["num_of_Convs"])):
          convolved_input = convolved
          if self.Params["UseBatchNormalization1"]:
            convolved_input = tf.layers.batch_normalization(
                convolved_input,
                training=self.is_training)
          # Add dropout layer if enabled and not first convolution layer.
          if i > 0 and (self.Params["Dropout1"]):
            convolved_input = tf.layers.dropout(
                convolved_input,
                rate=self.Params["dropoutval"],
                training=self.is_training,
                seed = self.seed)
          convolved = tf.layers.conv1d(
              convolved_input,
              filters=self.Params["num_of_Convs"][i],
              kernel_size=5,
              activation=tf.nn.relu,
              strides=1,
              padding="same",
              name="conv1d_%d" % i,
              kernel_initializer=tf.glorot_uniform_initializer(seed=(None if self.seed == None else self.seed+i+1)))
        return convolved
    
    def create_graph(self, GraphID):
        with self.Graphs[GraphID].as_default():
            with tf.variable_scope('Graph' + str(GraphID)):
                # PlaceHolders for inputs
                with tf.name_scope('Inputs'):
                    if self.Params["useEmbeddingLayer"]:
                        self.batch_ph.append(tf.placeholder(tf.int32, [None, self.seq_lens], name='batch_ph'))
                    else:
                        self.batch_ph.append(tf.placeholder(tf.float32, [None, self.seq_lens, self.vocabnum], name='batch_ph'))
                    self.seq_len_ph.append(tf.placeholder(tf.int32, [None], name='seq_len_ph'))
                    self.keep_prob_ph.append(tf.placeholder(tf.float32, name='keep_prob_ph'))
                    self.is_training.append(tf.placeholder(tf.bool, name='is_training'))
                    
                    #TODO: get indices instead of class embeddings themselves
                    self.CE.append(tf.placeholder(tf.float32, [None, self.ClassEmbeddingsize + 1], name="ClassEmbeddings"))
                    self.CKE.append(tf.placeholder(tf.float32, [None, self.ClassEmbeddingsize + 1], name="CandidateKinaseEmbeddings"))
                    self.TCI.append(tf.placeholder(tf.int32, [None], name="TrueClassIndices"))
                # Embedding layer
                if self.Params["useEmbeddingLayer"]:
                    with tf.name_scope('Embedding_layer'):
                        embeddings_var = tf.Variable(tf.random_uniform([self.vocabnum, self.Params["EMBEDDING_DIM"]], -1.0, 1.0, seed=self.seed[GraphID]), trainable=True)
                        #tf.summary.histogram('BiRNN/embeddings_var', embeddings_var)
                        batch_embedded = tf.nn.embedding_lookup(embeddings_var, self.batch_ph[GraphID])
                else:
                    batch_embedded = self.batch_ph[GraphID]
                if self.Params["rnn_unit_type"] != 'None':
                    if self.Params["UseBatchNormalization1"]:
                        batch_embedded = tf.layers.batch_normalization(
                            batch_embedded,
                            training=(self.is_training[GraphID]), name='batch_normalization_0_' + str(GraphID))
                    if len(self.Params["num_of_Convs"]) > 0:
                        with tf.name_scope('Conv_Layers'):
                            batch_embedded = self._add_conv_layers(batch_embedded)
                    
                    #tf.summary.histogram('BiRNN/batch_embedded', batch_embedded)
                    if self.Params["Dropout1"]:
                        batch_embedded = tf.nn.dropout(batch_embedded, keep_prob = self.keep_prob_ph[GraphID], seed=self.seed[GraphID], name='dropout_0_' + str(GraphID))
                    if self.Params["rnn_unit_type"] == 'CUDNNLSTM':
                        batch_embedded = tf.transpose(batch_embedded, [1, 0, 2])
                        lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
                                #input_size=self.seq_lens,
                                num_layers=self.Params["num_layers"],
                                num_units=self.Params["num_hidden_units"],
                                #dropout=self.dropoutval,
                                direction="bidirectional",
                                kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed[GraphID]),
                                name='Cudnn_'+str(GraphID))
                                #seed=self.seed)
                                #seed=0)
                        rnn_outputs, _ = lstm(batch_embedded)
                        rnn_outputs = tf.transpose(rnn_outputs, [1, 0, 2])
                    elif self.Params["rnn_unit_type"] != 'None':
                        with tf.variable_scope('rnn', initializer=tf.glorot_uniform_initializer(seed=self.seed[GraphID])):
                            fw_cells = [self.rnn_cell(GraphID, 'fw', L) for L in range(self.Params["num_layers"])]
                            bw_cells = [self.rnn_cell(GraphID, 'bw', L) for L in range(self.Params["num_layers"])]
                            
                            #fw_initial_states = [fw_cell.zero_state(self.batch_size, tf.float32) for fw_cell in fw_cells]
                            #bw_initial_states = [bw_cell.zero_state(self.batch_size, tf.float32) for bw_cell in bw_cells]
                            if self.Params["Bidirectional"]:
                                # (Bi-)RNN layer(-s)
                                rnn_outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=fw_cells, cells_bw=bw_cells, inputs=batch_embedded, sequence_length=self.seq_len_ph[GraphID], dtype=tf.float32, scope='stack_bidirectional_rnn_' + str(GraphID))#, initial_states_fw= fw_initial_states, initial_states_bw= bw_initial_states)
                                #rnn_outputs, _ = bi_rnn(self.rnn_cell(), self.rnn_cell(), inputs=batch_embedded, sequence_length=self.seq_len_ph, dtype=tf.float32)
                            else:
                                rnn_outputs, _ = tf.nn.dynamic_rnn(cell=self.rnn_cell(), inputs=batch_embedded, sequence_length=self.seq_len_ph[GraphID], dtype=tf.float32)
                    else:
                        rnn_outputs = batch_embedded
                    #tf.summary.histogram('BiRNN/RNN_outputs', rnn_outputs)
                    #tf.summary.tensor_summary('BiRNN/RNN_outputs', rnn_outputs)
                    if self.Params["UseBatchNormalization2"]:
                        rnn_outputs = tf.layers.batch_normalization(
                            rnn_outputs,
                            training=(self.is_training[GraphID]),
                            name='batch_normalization_1_' + str(GraphID))
                    
                    if self.Params["useAtt"]:
                        # Attention layer
                        with tf.name_scope('Attention_layer'):
                            attention_output, alphas = self.attention(rnn_outputs, self.Params["ATTENTION_SIZE"], GraphID, return_alphas=True)
                            self.alphas.append(alphas)
                            #tf.summary.histogram('BiRNN/alphas', self.alphas)
                    else:
                        attention_output = tf.concat(rnn_outputs, 2)
                        #self.attention_output = tf.reduce_sum(self.attention_output, axis=1)
                        attention_output = attention_output[:,-1,:]
                    if self.Params["IncreaseEmbSize"] > 0:
                        attention_output = tf.layers.dense(attention_output, IncreaseEmbSize, kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed[GraphID]))
            
                    # Dropout   
                    if self.Params["Dropout2"]:
                        attention_output = tf.nn.dropout(attention_output, keep_prob = self.keep_prob_ph[GraphID], seed=self.seed[GraphID], name='dropout_1_' + str(GraphID))
                    #self.embedding = tf.nn.l2_normalize(batch_embedded)
                    #self.embedding = tf.nn.l2_normalize(self.attention_output, axis=1)
                    #embedding = tf.pad(attention_output, [[0,0], [0,1]], constant_values=1, name='Pad_' + str(GraphID))
                
                else:
                    attention_output = tf.layers.Flatten()(batch_embedded)
                
                embedding = tf.pad(attention_output, [[0,0], [0,1]], constant_values=1, name='Pad_' + str(GraphID))
                
                if GraphID == 0:
                    self.emb = embedding
                
                with tf.name_scope('ZSL/Input'):
                    # tf Graph Input
                    DE = embedding
                    #tf.summary.histogram('Data_Embeddings_input', self.DE)
                    #tf.summary.tensor_summary('Data_Embeddings_input', self.DE)
                    
                
                with tf.name_scope('ZSL/Weights'):
                    # Define The W (model weights)
                    W = tf.Variable(tf.random_normal([DE.get_shape().as_list()[1], self.ClassEmbeddingsize + 1], mean=0.0, stddev=0.05, seed=self.seed[GraphID]), name="Weights_" + str(GraphID))
                    #tf.summary.histogram('Weights', W)
                    #tf.summary.tensor_summary('Weights', W)
                with tf.name_scope('ZSL/Calculate_Logits_'+str(GraphID)):
                    Matmul = tf.matmul(DE,W)
                    # Caclulate F = DE * W * CE for all the CEs in unique class embeddings (all the kinases)
                    logits = tf.matmul(Matmul, tf.transpose(self.CKE[GraphID]))
                    # Calculating the maximum of each row to normalize logits so that softmax doesn't overflow
                    maxlogits = tf.reduce_max(logits, 1, keep_dims=True)
                    #tf.summary.histogram('Output_logits', self.logits)
                    #tf.summary.tensor_summary('Output_logits', self.logits)
                    # Find the class index for each data point (the class with maximum F score)
                    outclassidx = tf.argmax(logits, 1)
                
                with tf.name_scope('ZSL/Calculate_Softmax_'+str(GraphID)):
                    denom = tf.reduce_sum(tf.exp(tf.subtract(logits, maxlogits)), 1)        
                    #denom = tf.reduce_sum(tf.exp(logits), 1)
                    M = tf.subtract(tf.reduce_sum(Matmul * self.CE[GraphID], 1), tf.squeeze(maxlogits))
                    #M = tf.reduce_sum(Matmul * self.CE[GraphID], 1)
                    rightprobs = tf.exp(M) / (denom + 1e-15) # Softmax
                    #self.rightprobs = tf.clip_by_value(self.rightprobs, clip_value_min= 1e-15, clip_value_max = 1 - 1e-15)
                    tf.summary.histogram('RightProbs', rightprobs)
                    #tf.summary.tensor_summary('RightProbs', rightprobs)
                
                #with tf.name_scope('ZSL/Accuracy'):
                    #self.accuracy, _ = tf.metrics.accuracy(predictions= self.outclassidx, labels = self.TCI)
                    #tf.summary.scalar('accuracy', self.accuracy)
                with tf.name_scope('ZSL/Loss_CrossEntropy_' + str(GraphID)):
                    # Calculate error using cross entropy
                    #self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.TCI))
                    #cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.TCI[GraphID]))
                    #vars = tf.trainable_variables() 
                    P = tf.clip_by_value(rightprobs, clip_value_min= 1e-15, clip_value_max = 1.1)
                    cost = tf.reduce_mean(-1 * tf.log(P))
                    #cost += tf.add_n([ tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name ]) * self.Params["regs"]
                    tf.summary.scalar('loss', cost)
                with tf.name_scope('ZSL/Optimizer_Metrics_'+str(GraphID)):
                    trainvars =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Graph' + str(GraphID))
                    # Gradient Descent
                    global_step = tf.Variable(0, trainable=False)
                    if self.Params["LRDecay"]:
                        rate = tf.train.exponential_decay(self.Params["learningrate"], global_step, 1, 0.99954, name="Learning_Rate")
                    else:
                        rate = tf.constant(self.Params["learningrate"])
                    tf.summary.scalar('Learning_Rate', rate)
                    optimizer = tf.contrib.layers.optimize_loss(loss=cost, global_step=global_step, learning_rate=rate, optimizer="Adam",
                                                                         # some gradient clipping stabilizes training in the beginning.
                                                                         clip_gradients=self.Params["ClippingGradients"]
                                                                         ,summaries=["learning_rate", "loss", "gradients", "gradient_norm"],
                                                                         variables=trainvars)
                
                merged = tf.summary.merge_all()
                
                self.train_writer['Graph_' + str(GraphID)] = tf.summary.FileWriter(os.path.join(self.LogDir,'Ensemble','logdir','Graph_' + str(GraphID), 'train'), optimizer.graph)
                self.test_writer['Graph_' + str(GraphID)] = tf.summary.FileWriter(os.path.join(self.LogDir,'Ensemble','logdir','Graph_' + str(GraphID), 'test'), optimizer.graph)
                self.val_writer['Graph_' + str(GraphID)] = tf.summary.FileWriter(os.path.join(self.LogDir,'Ensemble','logdir','Graph_' + str(GraphID), 'val'), optimizer.graph)
                
                
                #self.session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
            
            return {"optimizer": optimizer, "cost": cost, "outclassidx": outclassidx, "merged": merged, "W": W, "logits": logits, "rightprobs":rightprobs, "globalstep": global_step, "denom":denom, "M":M, "maxlogits":maxlogits, "logits":logits}
    def step(self, batch_X, batch_CE, batch_TCI, TrainCandidateKinases, Model, Modelind):
        seq_len = [self.seq_lens] * len(batch_X)
        #run_options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
        #self.run_metadata = tf.RunMetadata()
        #seq_len = np.array([list(x).index(-1) + 1 for x in batch_X])  # actual lengths of sequences
        input_feed = {self.batch_ph[Modelind]: batch_X,
                      self.CE[Modelind]: batch_CE,
                      self.CKE[Modelind]: TrainCandidateKinases,
                      self.TCI[Modelind]: batch_TCI,
                      self.seq_len_ph[Modelind]: seq_len,
                      self.keep_prob_ph[Modelind]: 1-self.Params["dropoutval"],
                      self.is_training[Modelind]: True}
        #Checking M and denom:
        #output_feed1 = [Model["M"], Model["denom"], Model["rightprobs"], Model["maxlogits"], Model["logits"]]        
        #outputs = self.sess[Modelind].run(output_feed1, input_feed)#, run_metadata=self.run_metadata, options=run_options)
        #print("\n\n========M : =======")
        #print(outputs[0])
        #print("\n\n========denom : =======")
        #print(outputs[1])
        #print("\n\n========rightprobs : =======")
        #print(outputs[2])
        #print("\n\n========maxlogits : ========")
        #print(outputs[3])
        #print("\n\n========logits : ========")
        #print(outputs[4])
        
        output_feed = [Model["optimizer"], Model["cost"], Model["outclassidx"], Model["merged"]]        
        outputs = self.sess[Modelind].run(output_feed, input_feed)#, run_metadata=self.run_metadata, options=run_options)
        return outputs
#    def next_batch(self, batch_size, DE, CE, TCI, index):
#        batch_DE = DE[index * batch_size:(index+1) * batch_size]
#        batch_CE = CE[index * batch_size:(index+1) * batch_size]
#        batch_TCI = TCI[index * batch_size:(index+1) * batch_size]
#        return batch_DE, batch_CE, batch_TCI
#        
    def next_batch(self, batch_size, DE, CE, TCI, modelindx, shuffle=True, FakeRand = False):
        """Taken From TensorFlow https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py"""
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch[modelindx]
        # Shuffle for the first epoch
        if self._epochs_completed[modelindx] == 0 and start == 0 and shuffle:
          perm0 = np.arange(len(DE))
          if FakeRand:
              np.random.RandomState(42).shuffle(perm0)
          else:
              np.random.shuffle(perm0)
          self.DE_Shuffled = DE[perm0]
          self.CE_Shuffled = CE[perm0]
          self.TCI_Shuffled = TCI[perm0]
        # Go to the next epoch
        if start + batch_size > len(DE):
          # Finished epoch
          self._epochs_completed[modelindx] += 1
          # Get the rest examples in this epoch
          rest_num_examples = len(DE) - start
          DE_rest_part = DE[start:len(DE)]
          CE_rest_part = CE[start:len(CE)]
          TCI_rest_part = TCI[start:len(TCI)]
          # Shuffle the data
          if shuffle:
            perm = np.arange(len(DE))
            if FakeRand:
                np.random.RandomState(17).shuffle(perm)
            else:
                np.random.shuffle(perm)
            self.DE_Shuffled = DE[perm]
            self.CE_Shuffled = CE[perm]
            self.TCI_Shuffled = TCI[perm]
          # Start next epoch
          start = 0
          self._index_in_epoch[modelindx] = batch_size - rest_num_examples
          end = self._index_in_epoch[modelindx]
          DE_new_part = self.DE_Shuffled[start:end]
          CE_new_part = self.CE_Shuffled[start:end]
          TCI_new_part = self.TCI_Shuffled[start:end]
          return np.concatenate(
              (DE_rest_part, DE_new_part), axis=0), np.concatenate(
                  (CE_rest_part, CE_new_part), axis=0), np.concatenate((TCI_rest_part, TCI_new_part), axis=0)
        else:
          self._index_in_epoch[modelindx] += batch_size
          end = self._index_in_epoch[modelindx]
          return self.DE_Shuffled[start:end], self.CE_Shuffled[start:end], self.TCI_Shuffled[start:end]
      
    def train(self, X, ClassEmbedding, TrainCandidateKinases, TrueClassIDX, epochcount, 
              ValDE = None, ValCandidatekinaseEmbeddings=None, ValCandidateKE_to_Kinase=None, ValKinaseUniProtIDs=None, ValKinaseEmbeddings=None, ValisMultiLabel=True, Val_TrueClassIDX=None, ValCandidateUniProtIDs=None,
              TestDE=None, TestCandidatekinaseEmbeddings=None, CandidateKE_to_Kinase=None, TestKinaseUniProtIDs=None, TestKinaseEmbeddings=None, TestisMultiLabel=True, Test_TrueClassIDX=None, TestCandidateUniProtIDs=None):
        """
        train the BiRNN model
        Args:
            X: Input sequence embeddings
            Y: One-hot encoded labels of kinases
            epochcount: how many epochs should the model run for
            X_test: sequence embeddings of validation data if provided the model will over this data
            Y_test = One-hot encoded labels of validation kinases
            Kinases = list of kinase class types for provided Y for training this is only for visualizing TSNE embeddings
        """
        print("Number of Train data: {} Number of Test data: {} Number of Val data: {}".format(len(X), len(TestDE), len(ValDE)))
        # Add one to the DataEmbeddings and ClassEmbeddings
        #DataEmbedding_with1 = np.c_[ DataEmbedding, np.ones(len(DataEmbedding)) ]
        ClassEmbedding_with1 = np.c_[ ClassEmbedding, np.ones(len(ClassEmbedding)) ]
        TrainCandidateKinases_with1 = np.c_[ TrainCandidateKinases, np.ones(len(TrainCandidateKinases)) ]
        # Find the indices of the true classes in the candidate class embeddings
        #TrueClassIDX = self.FindTrueClassIndices(ClassEmbedding_with1, TrainCandidateKinases_with1)
        #Test_TrueClassIDX = self.FindTrueClassIndices(TestKinaseEmbeddings, TestCandidatekinaseEmbeddings, TestisMultiLabel)
        #Val_TrueClassIDX = self.FindTrueClassIndices(ValKinaseEmbeddings, ValCandidatekinaseEmbeddings, ValisMultiLabel)
        self.training_epochs = epochcount
        
        self.num_examples = len(X)
        
        if TestKinaseUniProtIDs is not None:
            mlb_test = MultiLabelBinarizer()
            binlabels_true_test = mlb_test.fit_transform(TestKinaseUniProtIDs)
        if ValKinaseUniProtIDs is not None:
            mlb_Val = MultiLabelBinarizer()
            binlabels_true_Val = mlb_Val.fit_transform(ValKinaseUniProtIDs)
        
        Bestaccuracy_Val = 0
        Bestaccuracy_Test = 0
        Bestaccuracy_Train = 0
        Best_loss = 0
        
        Allresults = open(os.path.join(self.LogDir,'Allresults.csv'), 'w+')
        AllAccuracyTrains = np.zeros(self.Params["NumofModels"])
        AllAccuracyLoss = np.zeros(self.Params["NumofModels"])
        AllAccuracyVals = np.zeros(self.Params["NumofModels"])
        AllAccuracyTests = np.zeros(self.Params["NumofModels"])
        print("Epoch,," + ','.join(['TrainAcc_{}'.format(i) for i in range(self.Params["NumofModels"])]) + ',,' + ','.join(['ValAcc_{}'.format(i) for i in range(self.Params["NumofModels"])])+ ',,' + ','.join(['TestAcc_{}'.format(i) for i in range(self.Params["NumofModels"])]) + ',,' + 'ValAcc_Ensemble,TestAcc_Ensemble', file=Allresults)
        for epoch in range(self.training_epochs):
            #accuracy_train_avg_all = 0.0
            #loss_train_avg_all = 0.0
            print("===================================\nepoch: {}\t".format(self._epochs_completed))
            # Training
            num_batches = int(self.num_examples/self.Params["batch_size"]) + 1            
            start_time_allmodels = time.time()
            TestUniProtIDs = []
            ValUniProtIDs = []
            TestProbs = []
            ValProbs = []
            for i in range(self.Params["NumofModels"]):
                os.environ['PYTHONHASHSEED'] = '0'
                tf.set_random_seed(self.seed[i])
                np.random.seed(self.seed[i])
                random.seed(self.seed[i])
                #tf.set_random_seed(self.seed[i])
                loss_train = 0
                accuracy_train = 0
                print("============= Model: {} ===============".format(i))
                start_time_all = time.time()
                start_time_train = time.time()
                for b in range(num_batches):
                    batch_Xs, batch_CEs, batch_TCIs = self.next_batch(self.Params["batch_size"], X, ClassEmbedding_with1, TrueClassIDX, modelindx=i, FakeRand=False)#, batchnum=b, indices=idx)
                    _, c, OC, summary = self.step(batch_Xs, batch_CEs, batch_TCIs, TrainCandidateKinases_with1, self.Models[i], i)
                    #print("Global Step= " + str(GS))
                    
                    accuracy = accuracy_score(batch_TCIs, OC, normalize=True)
                    accuracy_train += accuracy
                    loss_train += c / num_batches
                duration_train = time.time()-start_time_train
                if ValDE is not None:
                    UniProtIDs, probabilities = self.predict(ValDE, ValCandidatekinaseEmbeddings, ValCandidateKE_to_Kinase, self.Models[i], ind=i, verbose=False)
                    UniProtIDs = UniProtIDs[0]
                    probabilities = probabilities[0]
                    ValUniProtIDs.append(UniProtIDs)
                    ValProbs.append(probabilities)
                    Val_summary= tf.Summary()
                    predlabels = [[label] for label in UniProtIDs]
                    binlabels_pred = mlb_Val.transform(predlabels)
                    if not os.path.exists(os.path.join(self.LogDir,'Ensemble','Reports_' + str(i))):
                        os.makedirs(os.path.join(self.LogDir,'Ensemble','Reports_' + str(i)))
                    Val_Evaluation = GetAccuracyMultiLabel(UniProtIDs, probabilities, ValKinaseUniProtIDs, Val_TrueClassIDX)
                    with open(os.path.join(self.LogDir,'Ensemble','Reports_' + str(i),'Val_'+str(epoch)+'.txt'), 'w+') as outfile:
                        print(classification_report(binlabels_true_Val, binlabels_pred, target_names=mlb_Val.classes_) + '\n\n\n' + 'Acccuracy_Val: {}  Loss_Val: {} Top5Accuracy: {} Top10Accuracy: {}'.format(Val_Evaluation["Accuracy"], Val_Evaluation["Loss"], Val_Evaluation["Top5Acc"], Val_Evaluation["Top10Acc"]), file=outfile)
                    Val_summary.value.add(tag="Accuracy", simple_value=Val_Evaluation["Accuracy"])
                    Val_summary.value.add(tag="Top5_Accuracy", simple_value=Val_Evaluation["Top5Acc"])
                    Val_summary.value.add(tag="Top10_Accuracy", simple_value=Val_Evaluation["Top10Acc"])
                    Val_summary.value.add(tag="Loss", simple_value=Val_Evaluation["Loss"])
                    self.val_writer['Graph_' + str(i)].add_summary(Val_summary, epoch)
                if TestDE is not None:
                    UniProtIDs, probabilities = self.predict(TestDE, TestCandidatekinaseEmbeddings, CandidateKE_to_Kinase, self.Models[i], ind=i, verbose=False)
                    UniProtIDs = UniProtIDs[0]
                    probabilities = probabilities[0]
                    TestUniProtIDs.append(UniProtIDs)
                    TestProbs.append(probabilities)
                    Test_summary= tf.Summary()
                    predlabels = [[label] for label in UniProtIDs]
                    binlabels_pred = mlb_test.transform(predlabels)
                    Test_Evaluation = GetAccuracyMultiLabel(UniProtIDs, probabilities, TestKinaseUniProtIDs, Test_TrueClassIDX)
                    with open(os.path.join(self.LogDir,'Ensemble','Reports_' + str(i),'Test_'+str(epoch)+'.txt'), 'w+') as outfile:
                        print(classification_report(binlabels_true_test, binlabels_pred, target_names=mlb_test.classes_)+ '\n\n\n' + 'Acccuracy_Test: {}  Loss_Test: {} Top5Accuracy: {} Top10Accuracy: {}'.format(Test_Evaluation["Accuracy"], Test_Evaluation["Loss"], Test_Evaluation["Top5Acc"], Test_Evaluation["Top10Acc"]), file=outfile)
                    Test_summary.value.add(tag="Accuracy", simple_value=Test_Evaluation["Accuracy"])
                    Test_summary.value.add(tag="Top5_Accuracy", simple_value=Test_Evaluation["Top5Acc"])
                    Test_summary.value.add(tag="Top10_Accuracy", simple_value=Test_Evaluation["Top10Acc"])
                    Test_summary.value.add(tag="Loss", simple_value=Test_Evaluation["Loss"])
                    self.test_writer['Graph_' + str(i)].add_summary(Test_summary, epoch)
                accuracy_train /= num_batches
                duration_all = time.time() - start_time_all
                print("train_loss: {:.3f}, train_acc: {:.3f}".format(loss_train, accuracy_train))
                print("Val_loss: {:.3f}, Val_acc: {:.3f}".format(Val_Evaluation["Loss"], Val_Evaluation["Accuracy"]))
                print("Test_loss: {:.3f}, Test_acc: {:.3f}".format(Test_Evaluation["Loss"], Val_Evaluation["Accuracy"]))
                print("Time_train: {}, Time_all: {}".format(duration_train, duration_all))
                
                AllAccuracyTrains[i] = accuracy_train
                AllAccuracyLoss[i] = loss_train
                AllAccuracyVals[i] = Val_Evaluation["Accuracy"]
                AllAccuracyTests[i] = Test_Evaluation["Accuracy"]
                
                Train_summary= tf.Summary()
                Train_summary.value.add(tag="Accuracy", simple_value=accuracy_train)
                Train_summary.value.add(tag="Loss", simple_value=loss_train)
                self.train_writer['Graph_' + str(i)].add_summary(Train_summary, epoch)
                self.train_writer['Graph_' + str(i)].add_summary(summary, epoch)
                
                if self.WriteEmbeddingVis:
                    self.VisualizeEmbedding(self.saver, self.train_writer, epoch, Kinases, X[:100])
                self.train_writer['Graph_' + str(i)].flush()
                if TestDE is not None:
                    self.test_writer['Graph_' + str(i)].flush()
            accuracy_train_ensemble = np.mean(AllAccuracyTrains)
            loss_train_ensemble = np.mean(AllAccuracyLoss)
            TestUniProtIDs, Testprobabilities = ensemble(TestUniProtIDs, TestProbs, TestCandidateUniProtIDs)
            Test_Evaluation = GetAccuracyMultiLabel(TestUniProtIDs, Testprobabilities, TestKinaseUniProtIDs, Test_TrueClassIDX)
            Test_summary= tf.Summary()
            Test_summary.value.add(tag="Accuracy", simple_value=Test_Evaluation["Accuracy"])
            Test_summary.value.add(tag="Top3_Accuracy", simple_value=Test_Evaluation["Top3Acc"])
            Test_summary.value.add(tag="Top5_Accuracy", simple_value=Test_Evaluation["Top5Acc"])
            Test_summary.value.add(tag="Top10_Accuracy", simple_value=Test_Evaluation["Top10Acc"])
            Test_summary.value.add(tag="Loss", simple_value=Test_Evaluation["Loss"])
            self.test_writer['Ensemble'].add_summary(Test_summary, epoch)
            
            if not os.path.exists(os.path.join(self.LogDir,'Ensemble','Reports_Ensemble')):
                os.makedirs(os.path.join(self.LogDir,'Ensemble','Reports_Ensemble'))
            predlabels = [[label] for label in TestUniProtIDs]
            binlabels_pred = mlb_test.transform(predlabels)
            with open(os.path.join(self.LogDir,'Ensemble','Reports_Ensemble','Test_'+str(epoch)+'.txt'), 'w+') as outfile:
                print(classification_report(binlabels_true_test, binlabels_pred, target_names=mlb_test.classes_)+ '\n\n\n' + 'Acccuracy_Test: {}  Loss_Test: {} Top5Accuracy: {} Top10Accuracy: {}'.format(Test_Evaluation["Accuracy"], Test_Evaluation["Loss"], Test_Evaluation["Top5Acc"], Test_Evaluation["Top10Acc"]), file=outfile)
                        
            ValUniProtIDs, Valprobabilities = ensemble(ValUniProtIDs, ValProbs, ValCandidateUniProtIDs)
            Val_Evaluation = GetAccuracyMultiLabel(ValUniProtIDs, Valprobabilities, ValKinaseUniProtIDs, Val_TrueClassIDX)
            Val_summary.value.add(tag="Accuracy", simple_value=Val_Evaluation["Accuracy"])
            Val_summary.value.add(tag="Top3", simple_value=Val_Evaluation["Top3Acc"])
            Val_summary.value.add(tag="Top5", simple_value=Val_Evaluation["Top5Acc"])
            Val_summary.value.add(tag="Top10_Accuracy", simple_value=Val_Evaluation["Top10Acc"])
            Val_summary.value.add(tag="Loss", simple_value=Val_Evaluation["Loss"])
            self.val_writer['Ensemble'].add_summary(Val_summary, epoch)
            print("\n\n Test Ensemble Accuracy: {:3f} Val Ensemble Accuracy: {:3f}".format(Test_Evaluation["Accuracy"], Val_Evaluation["Accuracy"]))
            
            print("{},,".format(epoch) + ','.join(['{}'.format(AllAccuracyTrains[i]) for i in range(self.Params["NumofModels"])]) + ',,' + ','.join(['{}'.format(AllAccuracyVals[i]) for i in range(self.Params["NumofModels"])])+ ',,' + ','.join(['{}'.format(AllAccuracyTests[i]) for i in range(self.Params["NumofModels"])]) + ',,' + '{},{}'.format(Val_Evaluation["Accuracy"],Test_Evaluation["Accuracy"]), file=Allresults)
            
            predlabels = [[label] for label in ValUniProtIDs]
            binlabels_pred = mlb_Val.transform(predlabels)
            with open(os.path.join(self.LogDir,'Ensemble','Reports_Ensemble','Val_'+str(epoch)+'.txt'), 'w+') as outfile:
                print(classification_report(binlabels_true_Val, binlabels_pred, target_names=mlb_Val.classes_) + '\n\n\n' + 'Acccuracy_Val: {}  Loss_Val: {} Top5Accuracy: {} Top10Accuracy: {}'.format(Val_Evaluation["Accuracy"], Val_Evaluation["Loss"], Val_Evaluation["Top5Acc"], Val_Evaluation["Top10Acc"]), file=outfile)
            
            self.test_writer['Ensemble'].flush()
            self.val_writer['Ensemble'].flush()
            Allresults.flush()
            if Val_Evaluation["Accuracy"] >= Bestaccuracy_Val:
                Bestaccuracy_Val = Val_Evaluation["Accuracy"]
                Bestaccuracy_Test = Test_Evaluation["Accuracy"]
                Bestaccuracy_Train = accuracy_train_ensemble
                Best_loss = loss_train_ensemble
                self.savemodel(Bestaccuracy_Train, Best_loss)
            print("Best accuracy Val: {}, Best accuracy Test: {}, BestAccTrain: {}, BestLossTrain: {}".format(Bestaccuracy_Val, Bestaccuracy_Test, Bestaccuracy_Train, Best_loss))
                
            duration_allMethods = time.time() - start_time_allmodels
            print("\nTime_all_Methods: {}".format(duration_allMethods))
        print("RNN Optimization Finished!")
        for i in range(self.Params["NumofModels"]):
            self.train_writer['Graph_' + str(i)].flush()
            self.train_writer['Graph_' + str(i)].close()
            if TestDE is not None:
                self.test_writer['Graph_' + str(i)].flush()
                self.test_writer['Graph_' + str(i)].close()
            if ValDE is not None:
                self.val_writer['Graph_' + str(i)].flush()
                self.val_writer['Graph_' + str(i)].close()
        self.test_writer['Ensemble'].flush()
        self.test_writer['Ensemble'].close()
        self.val_writer['Ensemble'].flush()
        self.val_writer['Ensemble'].close()
        Allresults.close()
        return Bestaccuracy_Train, Best_loss
    
    def predict(self,  DataEmbedding, TestCandidateKinases, CandidateKE_to_Kinase, Model = None, ind=None, verbose= True):
        """
        The method to predict the classes of given DataEmbeddings
        Args:
            DataEmbedding: The sequence embedding of the input kinases
            TestCandidateKinases: The list of candidate kinases
            verbose: Should the program write the Weight matrix in a file
        """
        # Add 1 to the end of Data embeddings and candidate kinase embeddings
        #DataEmbedding = np.c_[ DataEmbedding, np.ones(len(DataEmbedding)) ]
        if Model is not None:
            Models = [Model]
        else:
            Models = self.Models
        allUniProtIDs = []
        allprobs = []
        index = 0
        for idx, model in enumerate(Models):
            if ind is None:
                index = idx
            else:
                index = ind
            TestCandidateKinases_with1 = np.c_[ TestCandidateKinases, np.ones(len(TestCandidateKinases)) ]
            seq_len = [self.seq_lens] * len(DataEmbedding)
            
            input_feed = {self.batch_ph[index]: DataEmbedding,
                          self.CKE[index]: TestCandidateKinases_with1,
                          self.seq_len_ph[index]: seq_len,
                          self.keep_prob_ph[index]: 1,
                          self.is_training[index]: False}
            
            logits, W, OC = self.sess[index].run([model["logits"], model["W"], model["outclassidx"]], feed_dict=input_feed)
            if verbose:
                print("Writing weight matrix W in", os.path.join('EndToEndZSLWeights','ZSL_Weights' + str(self.seed) + '.txt'))
                np.savetxt(os.path.join('ZSLWeights','ZSL_Weights' + str(self.seed[idx]) + '.txt'), W)
           
            
            classes = TestCandidateKinases[OC]
            # get UniProtIDs for predicted classes and return them
            UniProtIDs =[]
            for KE in classes:
                UniProtIDs.append(CandidateKE_to_Kinase[tuple(KE)])
            UniProtIDs = np.array(UniProtIDs)
            allUniProtIDs.append(UniProtIDs)
            probabilities = self.softmax(logits, axis =1)
            allprobs.append(probabilities)
            
        return allUniProtIDs, allprobs
    
    def savemodel(self, TrainAcc, TrainLoss):
        self.ModelSaved = True
        for GraphID in range(self.Params["NumofModels"]):
            ckpt_dir = os.path.join(self.ckpt_dir, 'Graph_{}'.format(GraphID))
            if not os.path.isdir(ckpt_dir):
                os.makedirs(ckpt_dir)
            checkpoint_path = os.path.join(ckpt_dir, "model.ckpt")
            print("Saving the model...")
            self.saver[GraphID].save(self.sess[GraphID], checkpoint_path)
            with open(os.path.join(ckpt_dir,'EpochsCompleted.txt'), 'w+') as f:
                f.write('%d' % self._epochs_completed[GraphID])
            with open(os.path.join(ckpt_dir,'TrainAccAndLoss.txt'), 'w+') as f:
                f.write('%f,%f' % (TrainAcc, TrainLoss))
    def loadmodel(self):
        for GraphID in range(self.Params["NumofModels"]):
            ckpt_dir = os.path.join(self.ckpt_dir, 'Graph_{}'.format(GraphID))
            print("Restoring old model parameters from ", os.path.join(ckpt_dir, "model.ckpt"))
            self.saver[GraphID].restore(self.sess[GraphID], os.path.join(ckpt_dir, "model.ckpt"))
            with open(os.path.join(ckpt_dir,'EpochsCompleted.txt'), 'r') as f:
                for line in f:
                    self._epochs_completed[GraphID] = int(line)
            with open(os.path.join(ckpt_dir,'TrainAccAndLoss.txt'), 'r') as f:
                for line in f:
                    trainAcc = float(line.split(',')[0])
                    trainLoss = float(line.split(',')[1])
        return trainAcc, trainLoss, self._epochs_completed
        
    def FolderContainsModel(self):
        return os.path.isfile( os.path.join(self.ckpt_dir, "model.ckpt-0.meta")) 
        
    def initilize(self):
        self._index_in_epoch = np.zeros(self.Params["NumofModels"], dtype=int)
        self._epochs_completed = np.zeros(self.Params["NumofModels"], dtype=int)
        # Run the initializer
        for i in range(self.Params["NumofModels"]):
            with self.Graphs[i].as_default():
                init = tf.group(tf.global_variables_initializer())
                self.sess[i].run(init)
        
    def getEmbeddings(self, X):
        seq_len = [self.seq_lens] * self.Params["batch_size"]
        num_batches = int(len(X)/self.Params["batch_size"])
        EmbeddedXs = []
        #seq_len = [self.seq_lens] * len(X)
        for b in range(num_batches):
            batch_Xs = X[b * self.Params["batch_size"]: (b+1) * self.Params["batch_size"]]
            input_feed = {self.batch_ph[0]: batch_Xs,
                              self.seq_len_ph[0]: seq_len,
                              self.keep_prob_ph[0]: 1,
                              self.is_training[0]: False}
    
            output_feed = [self.emb]
            attention_output = self.sess[0].run(output_feed, input_feed)[0]
            if EmbeddedXs == []:
                EmbeddedXs = attention_output
            else:
                EmbeddedXs = np.concatenate((EmbeddedXs,attention_output),axis=0)
        if len(X) - (num_batches * self.Params["batch_size"]) > 0:
            batch_Xs = X[(num_batches * self.Params["batch_size"]):len(X)]
            seq_len = [self.seq_lens] * len(batch_Xs)
            input_feed = {self.batch_ph[0]: batch_Xs,
                              self.seq_len_ph[0]: seq_len,
                              self.keep_prob_ph[0]: 1,
                              self.is_training[0]: False}

            output_feed = [self.emb]
            attention_output = self.sess[0].run(output_feed, input_feed)[0]
            if EmbeddedXs == []:
                EmbeddedXs = attention_output
            else:
                EmbeddedXs = np.concatenate((EmbeddedXs,attention_output),axis=0)
        return EmbeddedXs
    
    def getEmbeddingSize(self):
        """
        Get the size of the final produced embedding by the model
        
        Returns:
            An integer showing the size of final embedding
        """
        shape = self.embedding.get_shape().as_list()
        return shape[-1]
    
    def getAttention(self, X):
        """
        Get attention values for given Xs
        
        Returns:
            Attention values for the given inputs
        """
        #seq_len = [self.seq_lens] * self.Params["batch_size"]
        num_batches = int(len(X)/self.Params["batch_size"])
        all_Attentions = []
        for i in range(self.Params["NumofModels"]):
            Attentions = []
            #seq_len = [self.seq_lens] * len(X)
            for b in range(num_batches):
                batch_Xs = X[b * self.Params["batch_size"]: (b+1) * self.Params["batch_size"]]
                seq_len = [self.seq_lens] * len(batch_Xs)
                input_feed = {self.batch_ph[i]: batch_Xs,
                                  self.seq_len_ph[i]: seq_len,
                                  self.keep_prob_ph[i]: 1,
                                  self.is_training[i]: False}
        
                output_feed = [self.alphas[i]]
                Attention = self.sess[i].run(output_feed, input_feed)[0]
                if Attentions == []:
                    Attentions = Attention
                else:
                    Attentions = np.concatenate((Attentions,Attention),axis=0)
            if len(X) - (num_batches * self.Params["batch_size"]) > 0:
                batch_Xs = X[(num_batches * self.Params["batch_size"]):len(X)]
                seq_len = [self.seq_lens] * len(batch_Xs)
                input_feed = {self.batch_ph[i]: batch_Xs,
                                  self.seq_len_ph[i]: seq_len,
                                  self.keep_prob_ph[i]: 1,
                                  self.is_training[i]: False}
    
                output_feed = [self.alphas[i]]
                Attention = self.sess[i].run(output_feed, input_feed)[0]
                if Attentions == []:
                    Attentions = Attention
                else:
                    Attentions = np.concatenate((Attentions,Attention),axis=0)
            all_Attentions.append(Attentions)
            #output_feed = [self.alphas]
            #Attention = self.sess.run(output_feed,input_feed)
        return all_Attentions
    
    def attention(self,inputs, attention_size, GraphID, time_major=False, return_alphas=False):
        """
        Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
    
        The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
         for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
        Variables notation is also inherited from the article
        
        Args:
            inputs: The Attention inputs.
                Matches outputs of RNN/Bi-RNN layer (not final state):
                    In case of RNN, this must be RNN outputs `Tensor`:
                        If time_major == False (default), this must be a tensor of shape:
                            `[batch_size, max_time, cell.output_size]`.
                        If time_major == True, this must be a tensor of shape:
                            `[max_time, batch_size, cell.output_size]`.
                    In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                    the backward RNN outputs `Tensor`.
                        If time_major == False (default),
                            outputs_fw is a `Tensor` shaped:
                            `[batch_size, max_time, cell_fw.output_size]`
                            and outputs_bw is a `Tensor` shaped:
                            `[batch_size, max_time, cell_bw.output_size]`.
                        If time_major == True,
                            outputs_fw is a `Tensor` shaped:
                            `[max_time, batch_size, cell_fw.output_size]`
                            and outputs_bw is a `Tensor` shaped:
                            `[max_time, batch_size, cell_bw.output_size]`.
            attention_size: Linear size of the Attention weights.
            time_major: The shape format of the `inputs` Tensors.
                If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
                If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
                Using `time_major = True` is a bit more efficient because it avoids
                transposes at the beginning and end of the RNN calculation.  However,
                most TensorFlow data is batch-major, so by default this function
                accepts input and emits output in batch-major form.
            return_alphas: Whether to return attention coefficients variable along with layer's output.
                Used for visualization purpose.
        Returns:
            The Attention output `Tensor`.
            In case of RNN, this will be a `Tensor` shaped:
                `[batch_size, cell.output_size]`.
            In case of Bidirectional RNN, this will be a `Tensor` shaped:
                `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
        """
    
        if isinstance(inputs, tuple):
            # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
            inputs = tf.concat(inputs, 2)
    
        if time_major:
            # (T,B,D) => (B,T,D)
            inputs = tf.array_ops.transpose(inputs, [1, 0, 2])
    
        hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer
    
        # Trainable parameters
        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.05, seed=self.seed[GraphID]))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.05, seed=self.seed[GraphID]))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.05, seed=self.seed[GraphID]))
    
        with tf.name_scope('v'):
            # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
            #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
            v.set_shape((inputs.shape[0],inputs.shape[1],attention_size))
    
        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        vu.set_shape((inputs.shape[0], inputs.shape[1]))
        alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape
        alphas.set_shape((inputs.shape[0], inputs.shape[1]))
        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
    
        if not return_alphas:
            return output
        else:
            return output, alphas
    def VisualizeEmbedding(self, saver, writer, step, Kinases, X):
        print("Writing Embeddings...")
        #sess.run(tf.global_variables_initializer())

        config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
        embedding_config = config.embeddings.add()
        embedding_config.tensor_name = self.embedding.name
        metadata_file = open(os.path.join(self.LogDir, 'metadataclasses_' + str(step) + '.tsv'), 'w')
        metadata_file.write('UniProtIDs\tNames\n')
        for i in range(len(X)):
            metadata_file.write('{}\t{}\n'.format(Kinases[i].UniprotID,Kinases[i].Protein_Name))
        metadata_file.close()
        embedding_config.metadata_path = os.path.join(self.LogDir, 'metadataclasses_' + str(step) + '.tsv')
        tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)
        #sess.run(self.embedding, feed_dict={x: mnist.test.images[:1024], y: mnist.test.labels[:1024]})
        self.getEmbeddings(X)
        saver.save(self.sess, os.path.join(self.LogDir,'BiRNN','logdir','train', "model.ckpt"), step)
    
    def softmax(self, X, theta = 1.0, axis = None):
        """
        Compute the softmax of each element along an axis of X.
    
        Parameters
        ----------
        X: ND-Array. Probably should be floats. 
        theta (optional): float parameter, used as a multiplier
            prior to exponentiation. Default = 1.0
        axis (optional): axis to compute values along. Default is the 
            first non-singleton axis.
    
        Returns an array the same size as X. The result will sum to 1
        along the specified axis.
        """
    
        # make X at least 2d
        y = np.atleast_2d(X)
    
        # find axis
        if axis is None:
            axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    
        # multiply y against the theta parameter, 
        y = y * float(theta)
    
        # subtract the max for numerical stability
        y = y - np.expand_dims(np.max(y, axis = axis), axis)
        
        # exponentiate y
        y = np.exp(y)
    
        # take the sum along the specified axis
        ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    
        # finally: divide elementwise
        p = y / ax_sum
    
        # flatten if X was 1D
        if len(X.shape) == 1: p = p.flatten()
    
        return p
    
    
    

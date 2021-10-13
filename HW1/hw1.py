#######
# Copyright 2020 Jian Zhang, All rights reserved
##

'''
Task 3:
-The average score obtained is:
    Average score: 0.49938385397195817
-The major difficulty for this GAN could be the diminishing gradient problem.
    Since the score scale is learned either with a 0 for ramdom sequences or 1 for 
    real music, the descriminator could get "too succesfull".
-Improving the way teh descriminator data set is sampled, with a more detailed scale
    May hel overcome this problem. 
'''
import logging
from abc import ABC, abstractmethod
import torch #pytorch
import torch.nn as nn
from torch.autograd import Variable 

import random
from sklearn.preprocessing import MinMaxScaler


class ModelBase(ABC):
    @abstractmethod
    def __init__(self, load_trained=False):
        '''
        :param load_trained
            If load_trained is True, load a trained model from a file.
            Should include code to download the file from Google drive if necessary.
            else, construct the model
        '''
        if load_trained:
            logging.info('load model from file ...')
            model = torch.load('./critic.pth')
            model.eval()
            return model

    @abstractmethod
    def train(self, x):
        '''
        Train the model on one batch of data
        :param x: train data. For composer training, a single torch tensor will be given
        and for critic training, x will be a tuple of two tensors (data, label)
        :return: (mean) loss of the model on the batch
        '''
        pass

class ComposerBase(ModelBase):
    '''
    Class wrapper for a model that can be trained to generate music sequences.
    '''
    @abstractmethod
    def compose(self, n):
        '''
        Generate a music sequence
        :param n: length of the sequence to be generated
        :return: the generated sequence
        '''
        pass


class CriticBase(ModelBase):
    '''
    Class wrapper for a model that can be trained to criticize music sequences.
    '''
    @abstractmethod
    def score(self, x):
        '''
        Compute the score of a music sequence
        :param x: a music sequence
        :return: the score between 0 and 1 that reflects the quality of the music: the closer to 1, the better
        '''
        pass

class Composer(nn.Module,ComposerBase):
    def __init__(self, seq_length, num_classes=1, input_size=2, hidden_size=2, num_layers=1):
        super(Composer, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.sigmoid(out) #relu
        out = self.fc(out) #Final Output
        return out


    def score(self,x):
        return(self.forward(x))

    def compose(self, n):
        '''
        Generate a music sequence
        :param n: length of the sequence to be generated
        :return: the generated sequence
        '''
        x_pred = [random.uniform(0,1),random.uniform(0,1)]
        
        seq = []
        for i in range(n):
            x_pred = Variable(torch.Tensor(x_pred)) #converting to Tensors
            x_pred = torch.reshape(x_pred, (1,1,x_pred.shape[0]))
            note = self.score(x_pred)
            seq.append(float(note))
            x_pred = [float(x_pred[0,0,1]),float(note)]
        return(seq)


    def train(self,model,x,y,num_epochs=1000,learning_rate=0.001,file_name='./composer.pth'):

        criterion = torch.nn.MSELoss()    # mean-squared error for regression
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

        for epoch in range(num_epochs):
          outputs = model.forward(x) #forward pass
          optimizer.zero_grad() #caluclate the gradient, manually setting to 0
         
          # obtain the loss function
          loss = criterion(outputs, y)
         
          loss.backward() #calculates the loss of the loss function
         
          optimizer.step() #improve from loss, i.e backprop
          if epoch % 1 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 

        torch.save(model, file_name)

class Critic(nn.Module,CriticBase):
    def __init__(self, seq_length, num_classes=1, input_size=51, hidden_size=2, num_layers=1):
        super(Critic, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.sigmoid(out) #relu
        out = self.fc(out) #Final Output
        return out


    def score(self,x):
        return(self.forward(x))

    def train(self,model,x,y,num_epochs=1000,learning_rate=0.001,file_name='./critic.pth'):

        criterion = torch.nn.MSELoss()    # mean-squared error for regression
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

        for epoch in range(num_epochs):
          outputs = model.forward(x) #forward pass
          optimizer.zero_grad() #caluclate the gradient, manually setting to 0
         
          # obtain the loss function
          loss = criterion(outputs, y)
         
          loss.backward() #calculates the loss of the loss function
         
          optimizer.step() #improve from loss, i.e backprop
          if epoch % 1 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 

        torch.save(model, file_name)
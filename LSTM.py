
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import time
from Custom_Dataset_Maker_Training import Training_Dataset_Maker
from Custom_DataSet_Maker_Testing import  Testing_Dataset_Maker
from torch.utils.data import DataLoader
#-----------------------------------------------------------------------------------------------------------------------
'''Directly using np.genfromtxt() causes memory overflow, so using pandas to read the csv'''
#combined_cpu_usage = np.genfromtxt("data/timeseries_training_testing_final.csv",delimiter=",")
combined_cpu_usage = pd.read_csv("data/final_data_after_preprocessing.csv",header=None, index_col= None)

print(combined_cpu_usage.shape)
combined_cpu_usage = np.array(combined_cpu_usage)

print("Shape of samples in the whole data without removing the samples that hinder batch processing: ",combined_cpu_usage.shape) #1068288

combined_cpu_usage = combined_cpu_usage[:2499200]

print("Shape of samples in the whole data after removing the samples that hinder batch processing: ",combined_cpu_usage.shape) #1068200

#----------------------------------------------------------------------------------------------------------------------
#combined_cpu_usage = combined_cpu_usage[0:10000] #only for checking for now

# plt.plot(combined_cpu_usage,"g")
# plt.show()
#

batch_size, no_of_timesteps_per_sequence = 100, 100


'''Create a generator that returns batches of size
       no_of_seqs_per_batch x no_of_timesteps_per_sequence from array_of_characters.

       Arguments
       ---------
       array_of_characters: Array you want to make batches from
       no_of_seqs_per_batch: Batch size, the number of sequences per batch
       no_of_timesteps_per_sequence: Number of sequence steps per batch
    '''



class LSTM_model(nn.Module):
    
    
    def __init__(self, per_timestep_feature_dimension = 1, no_of_timesteps_per_sequence=100, no_of_hidden_nodes_in_each_lstm_cell=512, no_of_stacked_lstm_layers=2,
                               drop_prob=0.5, lr=0.001):
        super(LSTM_model,self).__init__() #for python2
        #super().__init__() #for python 3
        self.drop_prob = drop_prob
        self.no_of_stacked_lstm_layers = no_of_stacked_lstm_layers
        self.no_of_hidden_nodes_in_each_lstm_cell = no_of_hidden_nodes_in_each_lstm_cell
        self.lr = lr
        

        
        ## Define the LSTM
        self.lstm = nn.LSTM(per_timestep_feature_dimension, no_of_hidden_nodes_in_each_lstm_cell, no_of_stacked_lstm_layers,
                            dropout=drop_prob, batch_first=True)
        
        ## Define a dropout layer
        self.dropout = nn.Dropout(drop_prob)

        ## Define the final, fully-connected output layer
        self.fc = nn.Linear(no_of_hidden_nodes_in_each_lstm_cell, 1)
        # output the logit for each item

      
    
    def forward(self, x, hidden_and_cell_states):


        ''' Forward pass through the network. These inputs are x, and the hidden and cell states `hidden_and_cell_states`. '''
        
        #Get x, and the new hidden state (h, c) from the lstm

        x, (h, c) = self.lstm(x, hidden_and_cell_states) #op is  (seq_len, batch, num_directions * hidden_size)

        '''VVI'''

        x = x [:][-1][:]
        #print(x[:][-1].shape) #since 512 nodes, so 512 outputs for each time step, we will take only the last from this and thus 512 values
        #All samples in the batch but only last timestep op: 100, 512


        ## Pass x through the dropout layer
        x = self.dropout(x)#will have to be taken care using model.evaluate()

        
        '''Dont remove the comment below, will be helpful in the future'''
        # Stack up LSTM outputs using view
        #x = x.view(x.size()[0]*x.size()[1], self.no_of_hidden_nodes_in_each_lstm_cell)
        #print(x.shape) torch.Size([12800, 512]) #think in terms of rows = samples and columns = features
        #This reshaping properly preserves each individual sample. An example can be found at the end of the file
        
        #12800 samples work as a batch

        #In case you want to use just the last timestep's output for regression (e.g., time-series prediction), you do not have
        #to use the softmax after a fully-connected layer.

        x = self.fc(x) #this fc layer will get samples * features dimension of input for batch processing and give 1 single scalar value
        #print(x.shape)#should be 100, 1 for our time-series prediction

        
        # Return x and the hidden state (h, c)
        return x, (h, c)
    
    
    def predict(self, input, h=None, cuda=False, top_k=None):
        
        ''' Given a character, predict the next character.
        
            Returns the predicted character and the hidden state.
        '''
        
        #The output of our RNN is from a fully-connected layer and it outputs a distribution of next-character 
        #scores. To actually get the next character, we apply a softmax function, 
        #which gives us a probability distribution that we can then sample to predict the next character
        
        
        if cuda:
            self.cuda()
        else:
            self.cpu()
        
        if h is None:
            h = self.init_hidden(1) 
        
        x = np.array([input])
        #x = one_hot_encoder(x, len(self.character_vocabulary))
        
        # inputs = torch.from_numpy(x)
        
        if cuda:
            inputs = x.cuda()
        
        h = tuple([each.data for each in h])
        out, h = self.forward(inputs, h)

        p = F.softmax(out, dim=1).data
        
        if cuda:
            p = p.cpu()
        
        if top_k is None:
            top_ch = np.arange(len(self.character_vocabulary))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()
        
        p = p.numpy().squeeze()
        
        char = np.random.choice(top_ch, p=p/p.sum())
            
        return self.int2char[char], h
    
    def init_weights(self):
        ''' Initialize weights for fully connected layer '''
        initrange = 0.1
        
        # Set bias tensor to all zeros
        self.fc.bias.data.fill_(0)
        # FC weights as random uniform
        self.fc.weight.data.uniform_(-1, 1)
        
    def init_hidden(self, no_of_seqs_per_batch):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x no_of_seqs_per_batch x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        return (weight.new(self.no_of_stacked_lstm_layers, no_of_seqs_per_batch, self.no_of_hidden_nodes_in_each_lstm_cell).zero_(),
                weight.new(self.no_of_stacked_lstm_layers, no_of_seqs_per_batch, self.no_of_hidden_nodes_in_each_lstm_cell).zero_())
        


# ### A note on the `predict`  function
# 
# The output of our RNN is from a fully-connected layer and it outputs a **distribution of next-character scores**.
# 
# To actually get the next character, we apply a softmax function, which gives us a *probability* distribution that we can then sample to predict the next character.

# In[34]:


def train(network, combined_cpu_usage_data, epochs=10, no_of_seqs_per_batch=10, no_of_timesteps_per_sequence=50, lr=0.001, clip=5, val_frac=0.1, cuda=False, print_every=10):
    ''' Training a network 
    
        Arguments
        ---------
        
        network: LSTM_Character_Generation network
        data: text data to train the network
        epochs: Number of epochs to train
        no_of_seqs_per_batch: Number of mini-sequences per mini-batch, aka batch size
        no_of_timesteps_per_sequence: Number of character steps in a sequence
        lr: learning rate
        clip: gradient clipping
        val_frac: Fraction of data to hold out for validation
        cuda: Train with CUDA on a GPU
        print_every: Number of steps for printing training and validation loss
    
    '''



    # create training and validation data
    validation_index = int(len(combined_cpu_usage_data) * (1 - val_frac))

    training_data, validation_data = combined_cpu_usage_data[:validation_index], combined_cpu_usage_data[
                                                                        validation_index:]  # training and validation data separation


    print("Total samples in training set after removing the validation set: ",len(training_data))#961459
    print("Toral samples in validation set: ",len(validation_data))

    training_dataset = Training_Dataset_Maker(training_data)

    validation_dataset = Testing_Dataset_Maker(validation_data)

    '''Build custom dala loaders'''
    training_dataloader = DataLoader(training_dataset, batch_size = 100, shuffle=True, num_workers=1)

    testing_dataloader =  DataLoader(validation_dataset, batch_size = 1, shuffle= True, num_workers= 1)

    network.train()#training mode
    
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    
    criterion = nn.L1Loss()#the suitable loss for such continuous values : L1 L2
    

    if cuda:
        network.cuda()
    
    counter = 0

    start_time = time.time()

    for e in range(epochs):

        print("Epoch No: ",e)
        
        #Note that we will initialize the hidden state at the beginning of each epoch of training
        h = network.init_hidden(no_of_seqs_per_batch)

        training_losses = []
        
        for batch_num, batch_of_x_and_label in enumerate(training_dataloader):#(data, no_of_seqs_per_batch, no_of_timesteps_per_sequence):



            '''this batch_of_data should be of dimension 100, 100'''
            
            counter += 1

            features = batch_of_x_and_label[0]
            features = features.float()
            features = features.reshape((features.shape[0],features.shape[1],1))
            labels = batch_of_x_and_label[1]


            
            if cuda:
                inputs, targets = features.cuda(), labels.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            network.zero_grad()


            output, h = network.forward(inputs, h)

            targets = targets.view(100,1).float()#.type(torch.cuda.LongTensor)

            #print(output.shape,targets.shape)#torch.Size([100, 1]) torch.Size([100, 1])

            
            loss = criterion(output, targets)

            training_losses.append(loss)

            loss.backward()
            
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(network.parameters(), clip)

            optimizer.step()

        '''Check how well the model performs at the end of each epoch'''
        '''PyTorch data loader doesnt allow variable sizes, so create another model with the same weights, use a different Data loader with batch size = 1'''

        print("Training loss: ",np.mean(training_losses))

        #VVi-----------------We need a new model with the same weights as the training model but a different batch size

        test_network = LSTM_model(feature_space_dimension_per_timestep, no_of_hidden_nodes_in_each_lstm_cell=512,no_of_stacked_lstm_layers=2)

        if cuda:
            test_network.cuda()
        # -------------------------------------------------------------move model to the device

        #get the parameters of the training model onto the testing model
        torch.save(network.state_dict(), 'training_model_statistics.pth')

        saved_state_statistics_of_training_model = torch.load("training_model_statistics.pth")

        test_model_statistics_dictionary = test_network.state_dict()

        for key, value in saved_state_statistics_of_training_model.items():

            if key in test_model_statistics_dictionary:

                    test_model_statistics_dictionary.update({key: value})  # -------------------------------------
                    test_network.load_state_dict(test_model_statistics_dictionary)

        # Get validation loss

        validation_h = network.init_hidden(no_of_seqs_per_batch)

        validation_losses = []

        testing_iterations_in_validation = 0

        for batch_num, batch_of_x_and_label in enumerate(testing_dataloader):  # (data, no_of_seqs_per_batch, no_of_timesteps_per_sequence):

            testing_iterations_in_validation += 1

            '''this batch_of_data should be of dimension 1, 100'''

            features = batch_of_x_and_label[0]
            features = features.float()
            features = features.reshape((features.shape[0], features.shape[1], 1))
            labels = batch_of_x_and_label[1]

            if cuda:
                inputs, targets = features.cuda(), labels.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            validation_h = tuple([each.data for each in validation_h])

            network.zero_grad()

            output, validation_h = network.forward(inputs, validation_h)

            targets = targets.view(100, 1).float()  # .type(torch.cuda.LongTensor)

            # print(output.shape,targets.shape)#torch.Size([100, 1]) torch.Size([100, 1])

            loss = criterion(output, targets)

            validation_losses.append(loss)
        print("Testing loss: ",np.mean(validation_losses))
        current_time = time.time()
        print("Time taken for epoch no: ",e,": ",current_time-start_time)
        start_time = current_time





'''Starts here'''

if __name__ == "__main__":


    # Initialize the network
    feature_space_dimension_per_timestep = 1
    network = LSTM_model(feature_space_dimension_per_timestep, no_of_hidden_nodes_in_each_lstm_cell=512, no_of_stacked_lstm_layers=2)
    print(network)




    #set to training mode
    network.train()
    train(network, combined_cpu_usage, epochs= 10, no_of_seqs_per_batch=batch_size, no_of_timesteps_per_sequence=no_of_timesteps_per_sequence, lr=0.0001, cuda=True, print_every=10)


    # change the name, for saving multiple files
    model_name = 'lstm_epoch.network'
    checkpoint = {'no_of_hidden_nodes_in_each_lstm_cell': network.no_of_hidden_nodes_in_each_lstm_cell,
                  'no_of_stacked_lstm_layers': network.no_of_stacked_lstm_layers,
                  'state_dict': network.state_dict(),}
                  # 'tokens': network.character_vocabulary}

    with open(model_name, 'wb') as f:
        torch.save(checkpoint, f)








#-------------------------------___Self------------------------------------
a= np.array( [[[1,2,3],
             [4,5,6]],
             
             [[7,8,9],
             [10,11,12]],
             
             [[13,14,15],
             [16,17,18]],
              
              [[19,20,21],
             [22,23,24]]
            
            
            ])


b = a.reshape(8, 3)


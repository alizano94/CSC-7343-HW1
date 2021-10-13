from abc import ABC, abstractmethod

class Aircraft(ABC):
  
    @abstractmethod
    def fly(self):
        pass

    @abstractmethod
    def land(self):
        pass

class Jet(Aircraft):
    def fly(self):
        print("My jet is flying")

    def land(self):
        print("My jet has landed")


jet1 = Jet()
jet1.fly()
jet1.land()

num_epochs = 10 #1000 epochs
learning_rate = 0.001 #0.001 lr

input_size = 51 #number of features
hidden_size = 2 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers

num_classes = 1 #number of output classes 

lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, x_train_tensors_final.shape[1]) #our lstm class 

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate) 

#Training process
for epoch in range(num_epochs):
  outputs = lstm1.forward(x_train_tensors_final) #forward pass
  optimizer.zero_grad() #caluclate the gradient, manually setting to 0
 
  # obtain the loss function
  loss = criterion(outputs, y_train_tensors)
 
  loss.backward() #calculates the loss of the loss function
 
  optimizer.step() #improve from loss, i.e backprop
  if epoch % 1 == 0:
    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 

torch.save(lstm1, './model.pth')

rand_midi = random_piano()
rand_seq = piano2seq(rand_midi)
rand_seq =  rand_seq[0:51]

rand_seq = Variable(torch.Tensor(rand_seq)) #converting to Tensors
rand_seq = torch.reshape(rand_seq, (1,rand_seq.shape[0]))
rand_seq = mm.transform(rand_seq)
rand_seq = Variable(torch.Tensor(rand_seq)) #converting to Tensors
rand_seq = torch.reshape(rand_seq, (1,1,rand_seq.shape[1]))

print(rand_seq.shape)

seq_predict = lstm1(rand_seq)#forward pass
seq_predict = seq_predict.data.numpy() #numpy conversion

print(seq_predict)

##########################################
num_epochs = 10 #1000 epochs
learning_rate = 0.001 #0.001 lr

input_size = 51 #number of features
hidden_size = 2 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers

num_classes = 1 #number of output classes 

root = './'
midi_load = process_midi_seq(datadir=root)
train_size = int(0.8*midi_load.shape[0])
train_seq_data = midi_load[0:int(train_size),:]
test_seq_data = midi_load[int(train_size):,:]

train_labels = np.ones(train_seq_data.shape[0])
test_labels = np.ones(test_seq_data.shape[0])


for i in range(train_seq_data.shape[0]):
    rand_midi = random_piano()
    rand_seq = piano2seq(rand_midi)
    rand_seq =  rand_seq[0:51]
    train_seq_data = np.vstack((train_seq_data, rand_seq))
    train_labels = np.append(train_labels,0)

for i in range(test_seq_data.shape[0]):
    rand_midi = random_piano()
    rand_seq = piano2seq(rand_midi)
    rand_seq =  rand_seq[0:51]
    test_seq_data = np.vstack((test_seq_data, rand_seq))
    test_labels = np.append(test_labels,0)

mm = MinMaxScaler()

train_seq_data = mm.fit_transform(train_seq_data)
test_seq_data = mm.fit_transform(test_seq_data)

x_train_tensors = Variable(torch.Tensor(train_seq_data))
x_test_tensors = Variable(torch.Tensor(test_seq_data))

y_train_tensors = Variable(torch.Tensor(train_labels))
y_test_tensors = Variable(torch.Tensor(test_labels)) 

x_train_tensors_final = torch.reshape(x_train_tensors,
                                    (x_train_tensors.shape[0],
                                    1, x_train_tensors.shape[1]))
x_test_tensors_final = torch.reshape(x_test_tensors,
                                    (x_test_tensors.shape[0], 
                                    1, x_test_tensors.shape[1]))

crt = Critic(1)
crt.train(x_train_tensors)
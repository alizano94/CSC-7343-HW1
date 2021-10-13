from Helper import *
from hw1 import *

import numpy as np
from sklearn.preprocessing import MinMaxScaler

num_epochs = 100

###############Critic training##################################
root = './'
midi_load = process_midi_seq(datadir=root,n=1000)
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


print(x_train_tensors_final)
print(y_train_tensors)

print(x_train_tensors_final.shape)
crt = Critic(x_train_tensors_final.shape[2])
print(type(crt))

crt.train(crt,x_train_tensors_final,y_train_tensors,num_epochs=num_epochs) 


###############Composer training##################################
root = './'
x_size = 2
x_data = []
y_data = []
midi_load = process_midi_seq(datadir=root,n=1000)
for i in range(midi_load.shape[0]):
    for j in range(midi_load.shape[1]-x_size):
        x_seq = midi_load[i,j:j+x_size]
        y_seq = midi_load[i,j+x_size]
        x_data.append(x_seq)
        y_data.append(y_seq)

mm = MinMaxScaler()

y_data = np.reshape(y_data,(len(y_data),1))

x_data = mm.fit_transform(x_data)
y_data = mm.fit_transform(y_data)

x_train_tensor = Variable(torch.Tensor(x_data))
x_train_tensors_final = torch.reshape(x_train_tensor,
                                    (x_train_tensor.shape[0],
                                    1, x_train_tensor.shape[1]))

y_train_tensor = Variable(torch.Tensor(y_data))

print(x_train_tensor.shape)
print(y_train_tensor.shape)

comp = Composer(x_train_tensors_final.shape[1])
print(type(comp))
comp.train(comp,x_train_tensors_final,y_train_tensor,num_epochs=num_epochs)

##################Experiment##############################
score_sum = 0
for i in range(100):
    composition = comp.compose(51)
    composition = Variable(torch.Tensor(composition))
    composition = torch.reshape(composition, (1, 1, composition.shape[0]))
    score = crt.score(composition)
    score_sum += float(score[0,0])

print('Average score: '+str(score_sum/100))

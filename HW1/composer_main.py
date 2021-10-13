from Helper import *
from hw1 import *

import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler


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
comp.train(comp,x_train_tensors_final,y_train_tensor,num_epochs=100)

x_pred = [random.uniform(0,1),random.uniform(0,1)]
x_pred = Variable(torch.Tensor(x_pred)) #converting to Tensors
x_pred = torch.reshape(x_pred, (1,1,x_pred.shape[0]))
print(x_pred)

y_pred = comp.score(x_pred)#forward pass
y_pred = y_pred.data.numpy() #numpy conversion
print(y_pred)
print(int(mm.inverse_transform(y_pred)))

comp.compose(10)
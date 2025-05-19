import torch
import numpy as np

input_size = 4
hidden_size = 2

h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

input_data = np.array([[h, e,l,l,o],
                       [e,o,l,l,l],
                       [l,l,e,e,l]], dtype=np.float32)
input_data = torch.Tensor(input_data)
rnn = torch.nn.RNN(input_size, hidden_size)
outputs, _status = rnn(input_data)
print(outputs, _status)
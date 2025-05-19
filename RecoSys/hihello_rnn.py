import torch
import numpy as np
#
# char_set = ['h', 'i', 'e', 'l', 'o']
#
# input_size = len(char_set)
# hidden_size = len(char_set)
# learning_rate = 0.1
#
# x_data = [[0,1,0,2,3,3]]
# x_one_hot = [[[1,0,0,0,0,],
#               [0,1,0,0,0],
#               [1,0,0,0,0],
#               [0,0,1,0,0],
#               [0,0,0,1,0],
#               [0,0,0,1,0]]]
# y_data = [[1,0,2,3,3,4]]
#
# X = torch.FloatTensor(x_one_hot)
# Y = torch.LongTensor(y_data)
#
# rnn = torch.nn.RNN(input_size, hidden_size)
# outputs, _status = rnn(X)
# print(outputs)
sample = "if you want you"

char_set = list(set(sample))
char_dic = {c: i for i, c in enumerate(char_set)}

dic_size = len(char_dic)
hidden_size = len(char_dic)
learning_rate = 0.1

sample_idx = [char_dic[c] for c in sample]
x_data = [sample_idx[:-1]]
x_one_hot = [np.eye(dic_size)[x] for x in x_data]
y_data = [sample_idx[1:]]

X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data)

rnn = torch.nn.RNN(dic_size, hidden_size, batch_first=True)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), learning_rate)

for i in range(100):
    optimizer.zero_grad()
    outputs, _status = rnn(X)
    loss = criterion(outputs.view(-1, dic_size), Y.view(-1))
    loss.backward()
    optimizer.step()

    result = outputs.data.numpy().argmax(axis=2)
    result_str = ''.join([char_set[c] for c in np.squeeze(result)])
    print(i, "loss: ", loss.item(), "prediction: ", result,
          "true Y: ", y_data, "prediction str: ", result_str)
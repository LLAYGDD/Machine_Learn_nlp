from Datasetprocess_rnn import *
from RNN_Layers import *


"""
数据导入
"""
"""
数据导入
"""

(x_train, x_test,y_train,y_test)=cross_validation.train_test_split(subtrain,labels,test_size=0.95,stratify=labels)


x_train = x_train.reshape((len(x_train),10,10)).astype(np.float32)
x_test = x_test.reshape((len(x_test),10,10)).astype(np.float32)


x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
print("train_data:", x_train.size())
print("text_data:", x_test.size())


data1 = RNNdataprocess(x_train,y_train)
data2 = RNNdataprocess(x_test,y_test)

train_loader = DataLoader(dataset=data1, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=data2, batch_size=128, shuffle=True)


rnn=RNN()
print(rnn)

optimizer=torch.optim.Adam(rnn.parameters(),lr=0.01)
loss_func=nn.CrossEntropyLoss()

for eopchs in range(50):
    for step,(x,y) in enumerate(train_loader):
        b_x=Variable(x.view(-1,10,10))
        b_y=Variable(y)

        output=rnn(b_x)
        loss=loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%100==0:
            for a,b in test_loader:
                test_x=Variable(a)
                test_y=Variable(b)
                test_output=rnn(test_x)

                accuracy = torch.max(test_output,1)[1].numpy() == test_y.numpy()
                print('accuracy:\t',accuracy.mean())
                break








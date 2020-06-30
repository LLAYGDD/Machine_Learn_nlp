from Datasetprocess import *
from CNN_Layers import *
from Demo_one.Data_processing import *



(x_train, x_test,y_train,y_test)=cross_validation.train_test_split(subtrain,labels,test_size=0.95,stratify=labels)


x_train = x_train.reshape((len(x_train),1,10,10)).astype(np.float32)
x_test = x_test.reshape((len(x_test),1,10,10)).astype(np.float32)


x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
print("train_data:", x_train.size())
print("text_data:", x_test.size())


data1 = myDataset(x_train,y_train)
data2 = myDataset(x_test,y_test)

train_loader = DataLoader(dataset=data1, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=data2, batch_size=128, shuffle=True)

cnn=CNN()
print(cnn)


optimizer=torch.optim.Adam(cnn.parameters(),lr=0.01)
loss_func=nn.CrossEntropyLoss()

print(loss_func)





loss_count = []
for epoch in range(10):
    for i,(x,y) in enumerate(train_loader):
        batch_x = Variable(x) # torch.Size([128, 1, 28, 28])
        batch_y = Variable(y) # torch.Size([128])
        # 获取最后输出
        out = cnn(batch_x) # torch.Size([128,10])
        # 获取损失
        loss = loss_func(out,batch_y)
        # 使用优化器优化损失
        optimizer.zero_grad()  # 清空上一步残余更新参数值
        loss.backward() # 误差反向传播，计算参数更新值
        optimizer.step() # 将参数更新值施加到net的parmeters上
        if i % 100 == 0:
            for a,b in test_loader:
                test_x = Variable(a)
                test_y = Variable(b)
                out = cnn(test_x)

                accuracy = torch.max(out,1)[1].numpy() == test_y.numpy()
                print('accuracy:\t',accuracy.mean())
                break

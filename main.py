import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import DNN
from matplotlib_inline import backend_inline
backend_inline.set_matplotlib_formats('svg')
x1=torch.rand(100000,1)
x2=torch.rand(100000,1)
x3=torch.rand(100000,1)
y1=((x1+x2+x3)<1).float()
y2=(((x1+x2+x3)>=1) &( (x1+x2+x3)<=2)).float()
y3=((x1+x2+x3)>2).float()
data=torch.cat([x1,x2,x3,y1,y2,y3],axis=1)
data=data.to('cuda:0')
data=data[torch.randperm(data.size(0)),:]
train_data=data[:(int)(data.size(0)*0.7),:]
test_data=data[(int)(data.size(0)*0.3):,:]
model=DNN.DNN().to('cuda:0')
epochs=10000
learning_rate=0.55
loss_fn=nn.MSELoss()
losses=[]
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
X=train_data[:,:3]
Y=train_data[:,3:]
for epoch in range(epochs):
    Pred=model.forward(X)
    loss=loss_fn(Pred,Y)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
fig=plt.figure()
plt.plot(range(epochs),losses)
plt.xlabel='epochs'
plt.ylabel='loss'
plt.show()
X1=test_data[:,:3]
Y1=test_data[:,3:]
with torch.no_grad():
    Pred=model.forward(X1)
    print(Pred)
    print(torch.argmax(Pred,axis=1))
    ts1=torch.arange(0,70000)
    Pred[ts1,torch.argmax(Pred,axis=1)]=1
    print(Pred)
    Pred[Pred!=1]=0
    correct=torch.sum((Pred==Y1).all(1))
    total=Y1.size(0)
    print(Pred)
    print(f"测试集精准度为：{100*correct/total}%")
with torch.no_grad():
    Pred=model.forward(X)
    ts1 = torch.arange(0, 70000)
    Pred[ts1, torch.argmax(Pred, axis=1)] = 1
    Pred[Pred!=1]=0
    correct=torch.sum((Pred==Y).all(1))
    total=Y.size(0)
    print(f"训练集精准度为：{100*correct/total}%")
torch.save(model,'model.pth')
new_model=torch.load('model.pth',weights_only=False)
with torch.no_grad():
    Pred=new_model.forward(X1)
    ts1 = torch.arange(0, 70000)
    Pred[ts1, torch.argmax(Pred, axis=1)] = 1
    Pred[Pred!=1]=0
    correct=torch.sum((Pred==Y1).all(1))
    print(f"新模型测试集精准度为：{correct*100/Y1.size(0)}%")
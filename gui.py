import torch
model=torch.load('model.pth',weights_only=False)
X=(list)(map(int,input("请输入3个0-1的数字").split(' ')))
ts1=torch.tensor(X).float().to('cuda:0')
with torch.no_grad():
    Pred=model.forward(ts1)
    print(Pred)
    ts2=torch.argmax(Pred)
    Pred[ts2]=1
    Pred[Pred!=1]=0
    print(Pred)
    if(Pred[0]==1):
        print("<1")
    elif(Pred[1]==1):
        print(">=1 and <=2")
    else:
        print(">2")
import torch.nn as nn
class DNN(nn.Module):
    def __init__(self):
        super(DNN,self).__init__()
        self.net=nn.Sequential(
            nn.Linear(3,5),nn.ReLU(),
            nn.Linear(5,5),nn.ReLU(),
            nn.Linear(5,5),nn.ReLU(),
            nn.Linear(5,3)
        )
    def forward(self,x):
        y=self.net(x)
        return y
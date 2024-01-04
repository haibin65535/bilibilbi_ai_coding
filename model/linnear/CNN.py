import torch
import torch.nn as nn

# 使用 torch.nn.Module 模板类构建模型
class CNN(nn.Module):
    def __init__(self,args):
        super(CNN,self).__init__()

        hidden = args.hidden
        self.cov1 = nn.Conv2d(in_channels=args.num_channels,out_channels=hidden,kernel_size=5,padding=2,stride=1)
        self.cov2 = nn.Conv2d(in_channels=hidden,out_channels=hidden,kernel_size=5,padding=2,stride=1)
        self.cov3 = nn.Conv2d(in_channels=hidden,out_channels=hidden,kernel_size=5,padding=2,stride=1)
        self.cov4 = nn.Conv2d(in_channels=hidden,out_channels=hidden,kernel_size=5,padding=2,stride=1)
        # 32, 28, 28

        self.linler = nn.Linear(in_features= hidden*28*28,out_features=hidden)
        self.output = nn.Linear(hidden,args.num_class)

    #这里是覆盖的模板类中的函数 （事实上这是一个抽象函数，如果不重写是无法正常执行代码的） 
    def forward(self,x):
        out = {}
        x = torch.relu(self.cov1(x))
        out['cov1'] = x
        x = torch.relu(self.cov2(x))
        out['cov2'] = x
        x = torch.relu(self.cov3(x))
        out['cov3'] = x
        x = torch.relu(self.cov4(x)) 
        out['cov4'] = x

        x = torch.flatten(x,start_dim=1)
        x = torch.relu(self.linler(x))
        out['linler'] = x

        x = torch.log_softmax(self.output(x),dim=1)
        if self.training:
            return x
        else:
            return x,out

    

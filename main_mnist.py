from dataset.mnist import get_dataset
from util.optm import get_argparse
from model.linnear.CNN import CNN
from train_eval.base import train
from util.io import get_save_file

from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch

# 编写这个文件的好处在于可以在服务器集群上选择固定GPU进行训练
# CUDA_VISIBLE_DEVICES=0 python main_mnist.py 
# 当集群中只有少数几个GPU可以使用且非主GPU可使用时 即可通过这种方式指定GPU进行训练
# 特别的当算力充足时这样子就可以同时运行很多份程序，且代码编写极为简便

if __name__ == '__main__':
    args = get_argparse()
    train_set,test_set =  get_dataset(args)
    if args.model == 'CNN':
        model = CNN(args)
    else:
        raise RuntimeError('model set error')
    
    train_loader = DataLoader(train_set,batch_size=args.batch_size,shuffle=True,pin_memory=True) # num_workers 一般不指定当使用cmd运行时可以视情况而定
    test_loader = DataLoader(test_set,batch_size=4*args.batch_size,shuffle=False,pin_memory=True)    
    optm = Adam(model.parameters(),lr=1e-5) #lr 不宜过大
    creation = nn.NLLLoss() # 可以与log_softmax配合组成becloss  也就是二元交叉熵损失

    train(model,train_loader,test_loader,optm,creation,args)

    # 需要保存模型
    torch.save(model,get_save_file(args) + 'saving_model.pt')


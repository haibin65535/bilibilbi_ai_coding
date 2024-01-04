import argparse
import torch


# 获取整个任务中会使用到的超参数，这个超参数将贯穿整个任务执行流程
# 这个模块有一些弊端 jupyter 和cmd运行是需要不同的方式的

def get_argparse(TERMINAL_CMD = True):

    parser = argparse.ArgumentParser()


    # ------------------------------------dataset------------------------
    parser.add_argument('--noise', type=float,default= 0.4)
    parser.add_argument('--dataset', type=str,default='mnist')
    parser.add_argument('--num_class', type=int,default=10)   # 保存一下这个参数后续使用
    parser.add_argument('--num_channels', type=int,default=1)  # 上一集中看到的数据x的通道数 【1,28,28】中的 1 torch 框架下的大部分数据都是第一个维度大小就是这个数据的通道数

    #------------------------------------model---------------------------
    parser.add_argument('--model', type=str,default='CNN')
    parser.add_argument('--hidden', type=int, default=32)        # 设置隐藏层的通道数量
    
    #------------------------------------train---------------------------
    parser.add_argument('--epochs', type=int, default=10)        # 训练循环次数
    parser.add_argument('--batch_size', type=int, default=128)        # minibatch 大小 
    parser.add_argument('--save_data',action= 'store_true',default=False)        # minibatch 大小    
    
    args = parser.parse_args() if TERMINAL_CMD else parser.parse_args(args=[])# 给jupyter运行环境留出操作空间
    
    args.cuda = torch.cuda.is_available()
    return args
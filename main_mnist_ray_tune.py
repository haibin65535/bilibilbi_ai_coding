from dataset.mnist import get_dataset
from util.optm import get_argparse
from model.linnear.CNN import CNN
from train_eval.base import train


from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from ray import tune  
from functools import partial

def change_config_to_args(config:dict,args):
    for key in config.keys():
        setattr(args,key,config[key]) # 将ray框架生成的参数映射到我们自己的超参数当中
    return args

# 这次搜索演示 将直接进行顶层的超参数搜索 每次迭代为整个训练模型与评估模型（只评估准确率）的流程
def tune_main(config,args):
    args = change_config_to_args(config,args)
    
    train_set,test_set =  get_dataset(args)
    if args.model == 'CNN':
        model = CNN(args)
    else:
        raise RuntimeError('model set error')
    
    train_loader = DataLoader(train_set,batch_size=args.batch_size,shuffle=True,pin_memory=True) # num_workers 一般不指定当使用cmd运行时可以视情况而定
    test_loader = DataLoader(test_set,batch_size=4*args.batch_size,shuffle=False,pin_memory=True)    
    optm = Adam(model.parameters(),lr=1e-3) #lr 不宜过大
    creation = nn.NLLLoss() # 可以与log_softmax配合组成becloss  也就是二元交叉熵损失

    return train(model,train_loader,test_loader,optm,creation,args)

    

# 编写这个文件的好处在于可以在服务器集群上选择固定GPU进行训练
# CUDA_VISIBLE_DEVICES=0 python main_mnist.py 
# 当集群中只有少数几个GPU可以使用且非主GPU可使用时 即可通过这种方式指定GPU进行训练
# 特别的当算力充足时这样子就可以同时运行很多份程序，且代码编写极为简便

if __name__ == '__main__':
    args = get_argparse()  # 看来这个东西只能通过partial 传递进去 不然会出问题
    config = {
        'hidden':  tune.grid_search([16 , 32,64,128]), #这样子就是网格搜索          
        'epochs':  tune.grid_search([ 5 ,10 , 20, 30 , 40]), #这样子就是网格搜索      
        'batch_size': tune.grid_search([ 32 ,64 , 128, 256]) #这样子就是网格搜索
    }

    '''
    config = {
        'hidden': 16 , # tune.grid_search([16 , 32,64,128]), #这样子就是网格搜索          
        'epochs':  10 ,# tune.grid_search([ 5 ,10 , 20, 30 , 40]), #这样子就是网格搜索      
        'batch_size': 128 #  tune.grid_search([ 32 ,64 , 128, 256]), #这样子就是网格搜索
    }

    tune_main(config)
    
    '''
    analysis = tune.run(
            partial(tune_main,args=args),# 这个函数是只接受config参数的函数  可以使用partial 包达成这个效果但是不建议
            metric="mean_accuracy",
            mode= "max",
            name= 'tune_1',
            # scheduler=sched, 可以通过这个参数使用更加复杂的搜索算法
            resources_per_trial={"cpu": 1,"gpu": 1},
            num_samples = 1,
            config = config)



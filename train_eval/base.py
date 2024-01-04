import torch
from tqdm import tqdm
from util.io import save_as_numpy

def train_epoch(model,loader,optm,creation,args):
    model.train()
    for data,lable in tqdm(loader,desc='train'):
        
        if args.cuda:
            data = data.cuda()
            lable = lable.cuda()
        
        out_put = model(data)

        loss = creation(out_put,lable)

        optm.zero_grad()
        loss.backward()
        optm.step()

def eval(model,loader,creation,args,mode = 'train_data'):
    sum_loss = 0
    sum_acc = 0
    sum_lable = 0
    model.eval()
    for data,lable in tqdm(loader,desc='test:'+ mode):
        if args.cuda:
            data = data.cuda()
            lable = lable.cuda()

        out_put,_ = model(data)
        sum_loss += creation(out_put,lable).item()
        sum_acc += (torch.argmax(out_put,dim=1) == lable).sum()
        sum_lable += lable.shape[0] 

    return sum_loss,(sum_acc/sum_lable).item()

def train(model,train_loader,test_loader,optm,creation,args):
    epochs = args.epochs
    if args.cuda:
        model.cuda()
    
    train_losss,train_accs,test_losss,test_accs = [],[],[],[],

    for epoch in range(epochs):
        train_epoch(model,train_loader,optm,creation,args)
        
        train_loss,train_acc = eval(model,train_loader,creation,args)
        test_loss,test_acc = eval(model,test_loader,creation,args,mode='test_data')
        print('epoch : [{}/{}] | train : [loss : {:.4f} , acc : {:.4f}] |  test : [loss : {:.4f} , acc : {:.4f}]  '
              .format(epoch,epochs,
                      train_loss,train_acc,
                      test_loss,test_acc))
        
        train_losss.append(train_loss) 
        train_accs.append( train_acc ) 
        test_losss.append( test_loss ) 
        test_accs.append(  test_acc  )

    if args.save_data:
        save_as_numpy(train_losss,'train_losss',args)
        save_as_numpy(train_accs,'train_accs',args)
        save_as_numpy(test_losss,'test_losss',args)
        save_as_numpy(test_accs,'test_accs',args)
    
    return test_acc
    




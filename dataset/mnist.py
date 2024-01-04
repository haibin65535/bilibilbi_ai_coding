import torchvision
from torchvision import transforms
# @ args 超参数对象
# 这个函数将返回一个 拥有正常训练集 ，但是对测试集添加符合高斯分布的噪声的数据集当 noise为0时就会返回正常数据集了
def get_dataset(args):

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.7),std= (0.7))])


    #这里需要知道当前使用的是哪一个数据集

    dataset_name =  args.dataset.lower()

    # 然后根据数据集名称返回数据集
    # 当固定框架中没有需要使用的数据集时就可以直接在这个函数中以及它的上游进行操作


    if dataset_name  == 'mnist':
        train_set = torchvision.datasets.MNIST(root = 'dataset/data',train=True,download=True,transform=transform)
        test_set = torchvision.datasets.MNIST(root = 'dataset/data',train=False,download=True,transform=transform)

    elif dataset_name  == 'fashion':
        train_set = torchvision.datasets.FashionMNIST(root = 'dataset/data',train=True,download=True,transform=transform)
        test_set = torchvision.datasets.FashionMNIST(root = 'dataset/data',train=False,download=True,transform=transform)
    elif dataset_name == 'kuzu':
        train_set = torchvision.datasets.KMNIST(root = 'dataset/data',train=True,download=True,transform=transform)
        test_set = torchvision.datasets.KMNIST(root = 'dataset/data',train=False,download=True,transform=transform)
    else:
        raise RuntimeError ('only support [mnist ,fashion ,kuzu] ')



    import torch
    test_set.data = test_set.data.float()
    test_set.data += args.noise*torch.randn_like(test_set.data)
    
    return train_set,test_set    
    


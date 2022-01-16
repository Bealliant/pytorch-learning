_最近很emo，然后就很久都没有更新相关的内容了_
其实这一篇文章其实是[Pytorch深度学习实践by Prof. Hongpu LIU](https://www.bilibili.com/video/BV1Y7411d7Ys?p=11&spm_id_from=pageDriver)中的两课的合集，我们先学习了内置Dataset类型的实例化，然后再对于内置的Dataset MINIST数据集进行分类的操作。
# Dataset
```python
from torch.utils.data import Dataset
```
通过这个代码，我们其实import了一个名叫Datasets的抽象类。
之所以叫做抽象类，是因为这个Datasets不能够直接进行实例化，只能通过把它当成父类进行类的继承才可以。换而言之，我们需要定义自己的Dataset类，然后再继承他。
在课上，我们使用的是sklearn文件夹之中的一个Diabetes（糖尿病）的数据集进行操作。
对于自定义的数据集，需要对于这个类写入三个方法，分别是`def  __init__(self)`\ `def __getitem(self,args)__` \ `def __len__(self)`。
initialize方法定义每一个类都需要有的初始化，
get item 这个方法相当于对于每一个类可以使之能够被索引
length返回的是这个类的实例的长度
```python
class Data_diabetes(Dataset):
    def __init__(self):
        dataset_numpy = np.loadtxt("diabetes.csv.gz", delimiter=',', dtype=np.float32)
        self.length = dataset_numpy.shape[0]
        self.x = dataset_numpy[:, :-1]
        self.y = dataset_numpy[:, -1].reshape(self.length,1)

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return self.length
```
## `def __init__(self)`
事实上，读TXT文件的时候使用numpy是更好的选择。它和matlab之中的`load(filepath)`更加接近。
我们读了一个numpy类型的数据进来, 再通过`torch.from_numpy`方法把numpy类型转化成tensor类型。
剩下的几个参数：
**delimiter**: 数据之间的分隔符
**dtype** :这个很重要，torch进行操作的时候要求数据的类型是`torch.float`类型

## `def __getitem__(self,x)`

返回的是这个类的对象的索引值

## **注意事项**
如果读入的是自己定义的图像的话，在init操作之中读入的只是读入一个图片的list，而不是把所有的图片都打开。
然后再`__getitem__(self,x)`的操作之中，使用PIL中的Image模块打开一个列表的list[x]，然后对于这一张图片进行一些操作。比如说Normalize或者是ToTensor。

最后我们把这个定义的Diabetes类进行实例化，就得到了我们的实例化数据对象
```python
Diabetes_Dataset = Data_Diabetes() #根据init，初始化不需要额外的参数
```

# DataLoader
DataLoader是通过对于所有的datasets进行洗牌的形式，来提高训练的模型的泛化能力。
```python
Diabetes_loader = DataLoader(Diabetes_Dataset, batch_size = 32, shuffle = True)
```
一般对于dataloader的初始化也很简单，只要设定这几个参数就可以了
`dataset`
`batch_size` ： 传入的每一个batch的大小
`shuffle = True` ： 随机洗牌：ON，（**test_set 要设置Shuffle为OFF**）

## enumerate
对于list\tuple\Loader 这一种Iterable（可迭代）的对象，在for循环里面，使用enumerate可以返回每一个元素及其索引值。

然后使用我们之前已经有的方法就可以进行diabetes样本的训练了。

对于测试集的话就不需要进行打乱了.

# MINIST实战
采用了四个全连接层进行训练，激活函数为ReLU。
如果要输出多分类的结果的话，可以采用softmax函数，
$$
F_{(x)}=softmax({x_{i}}) = \frac{1}{\sum^{n}_{i=1} e ^{x_{i}}} (for\  \forall \ i \in [1,\ n])
$$
在pytorch之中有更加现成的接口，直接集成了交叉熵和Softmax函数
![torch.nn.CrossEntropyLoss](https://img-blog.csdnimg.cn/7f0c68e36d974c26a34010a22fef89de.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQmVhbGxpYW50,size_20,color_FFFFFF,t_70,g_se,x_16)
## transforms 零散操作
```python
from torchvision import transforms
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307),(0.3081))
])
```
这里本质上还是对于transforms中的几个类进行的实例化.
transform是torchvision之中对于图像进行操作的一个package.
`ToTensor`: 把PIL类型的图片转化成张量的形式，方便进行后续的神经网络处理.
`Normalize`:也叫做 Feature Scaling, 可以异步至吴恩达的网课[Andrew Ng ML特征缩放](https://www.bilibili.com/video/BV164411b7dx?p=20).
通过这一步操作把数据转换成[-1~+1]的范围,防止数据的范围对于训练的影响.

## MINIST数据的打开
`MINIST`和`CIFAR10`\ `COCO` 这一些知名的数据集一般都存放在`torchvision.datasets` 之中,使用之前要先导入.
```python
def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
```
查看他的源码我们可以发现,这几个参数比较重要:
`root` : 打开文件的目录(directory)
`train`: 是否是训练数据集(**注意:测试集一定要置OFF,否则测试就不会有意义了**)
`transform`: 对于图像的操作(传入一个Obj.)
`download`: 如果没有找到的话,是否进行下载?

于是我们新建了测试集\测试Loader\训练集\训练Loader
设置Batch_Size为32.
```python
train_dataset = datasets.MNIST(root="/MINIST",
							train=True,
							transform=transforms,download=True)
train_loader = DataLoader(train_dataset,
							batch_size=32,
							shuffle=True)
test_dataset = datasets.MNIST(root="/MINIST",
							train=False, 
							transform= transforms)
test_loader = datasets.MNIST(root="/MINIST",
							train=False,
							transform=transforms,
							download=True)
```

## 建立全连接神经网络
```python
class neural_network(torch.nn.Module):
    def __init__(self):
        super(neural_network, self).__init__()
        self.linear1 = torch.nn.Linear(784, 512)
        self.linear2 = torch.nn.Linear(512, 256)
        self.linear3 = torch.nn.Linear(256, 64)
        self.linear4 = torch.nn.Linear(64,10)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = x.view(-1,784)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        return x
```
**注意:**
>`x = x.view(-1,784)`

是对于整个(batch_size,1,28,28) 的Tensor对象进行一个Flatten操作.
**`-1`表示的是`自由计算`的意思**
这句话的意思是,把原来的Tensor对象x的最后一个维度拉平,变成-1,别的维度让机器自己算.
## Summary Writer 支持
>tensorboard原本是tensorflow的可视化工具，pytorch从1.2.0开始支持tensorboard。之前的版本也可以使用tensorboardX代替。
>
[PyTorch下的Tensorboard 使用](https://zhuanlan.zhihu.com/p/103630393)

记得之前也在B站跟着小土堆学过tensorboard和transforms,但全都忘光了(bushi)
查了好久,然后这次就只用了`add_scalar`的操作,画出`epoch`和`loss`\ `correct rate`之间的关系

## 实例化:
起飞前的准备.
对于神经网络`neural_network`,优化器`optimizer`,损失函数`loss`进行实例化.
这一次的tryout里面我们引入了tensorboard中的summary writer,所以这一次是四个:
```python
model = neural_network()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
writer = SummaryWriter()
```
表示和这四句话真的太熟了(bushi)

## `def train(epoch):`
通过这个函数把训练神经网络的过程进行封装.
因为在真正的训练的过程之中,我们会先train一遍,然后再评估一下这个模型的准确度(accuracy).这样的两个步骤就是一个epoch循环中的组成部分.
这样封装的好处是,让epoch循环可以以更加简洁的方式来表示.
```python
def train(epoch):
    sum = 0.0
    for batch_indx, data in enumerate(train_loader):
        x_data, y_data = data
        pred = model(x_data)
        loss = criterion(pred,y_data)
        sum+=loss
        print("\r"+"#"*epoch + str(epoch)+ '\t'+str(loss.item()),end = '')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    writer.add_scalar('loss-epoch',loss.item(),epoch)
```
可以看出,这个train过程的大致思路和之前的案例非常类似.
需要强调的是,add_scalar方法可以在图像之中加上一个点 $(epoch,loss_{epoch})$,从而成为一个图像.
## `def test(epoch):`
在这个过程之中有几个使用的新方法:
`torch.max(tensor,dim=0 )` 以`tensor`形式返回一个`tensor`类型中的最大值.(可以参考一下matlab中的最大值函数)
`dim`默认为0,就是以`行`为索引,`每一列的最大值`,当`dim=1`的时候,返回的是`列`的索引,`每一行的最大值`.
以`tuple`类型返回(`值 value`,`索引 idx`).
最后的预测值`pred`是一个`(n*10)`的表. 我们需要找到这个最大值对应的索引.
```python
_,predicted = torch.max(pred.data,dim=1)	#接收表中每一行最大值的索引
```
为了找到最后的正确率,需要找到正确的个数,以及总样本的个数.
```py
total+=x_data.size(0)
```
可以让我们通过循环求出样本的总容量.

## 逻辑数组(Logical Array)
在`MATLAB`之中,我们可以通过逻辑数组来返回矩阵中每个元素是否满足某个条件的布尔值.
比如说,在MATLAB命令提示行之中输入:
```Octave
>> A = [1,2,3,4; 5,6,7,8; 9,10,11,12];
>> A > 5
```
会得到这样的Output:
```Octave

ans =

  3×4 logical array

   0   0   0   0
   0   1   1   1
   1   1   1   1

>> 
```
这里返回的是Boolean类型,1表示满足,0表示不满足.

Pytorch之中也是同理,我们可以先得到这一个逻辑矩阵,然后对于逻辑矩阵进行求和,就得到了正确的个数.
(因为如果正确的话,返回的每一个数都是0.)

**注意: 返回的Logistic Array 仍然是一个Tensor类型的对象,需要使用**`.item()` 得到这个里面的值.

最后:

```python
writer.add_scalar('correct rate - epoch', correct/total, epoch)
```
就可以得到另外一个Graph,关于$(epoch, Correct\ Rate\ _{epoch})$

## `main`
```py
for epoch in range(100):
    train(epoch)
    test(epoch)
```
让训练的函数和测试的过程跑起来!


# `Source Code`
```python
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import time

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307),(0.3081))
])

train_dataset = datasets.MNIST(root="/MINIST",
								train=True,
								transform=transforms,
								download=True)
train_loader = DataLoader(train_dataset,
								batch_size=32,
								shuffle=True)
test_dataset = datasets.MNIST(root="/MINIST",
								train=False, 
								transform= transforms)
test_loader = datasets.MNIST(test_dataset,
								train=False,
								transform=transforms,
								download=True)

class neural_network(torch.nn.Module):
    def __init__(self):
        super(neural_network, self).__init__()
        self.linear1 = torch.nn.Linear(784, 512)
        self.linear2 = torch.nn.Linear(512, 256)
        self.linear3 = torch.nn.Linear(256, 64)
        self.linear4 = torch.nn.Linear(64,10)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = x.view(-1,784)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        return x


model = neural_network()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
writer = SummaryWriter()
def train(epoch):
    sum = 0.0
    for batch_indx, data in enumerate(train_loader):
        x_data, y_data = data
        pred = model(x_data)
        loss = criterion(pred,y_data)
        sum+=loss
        print("\r"+"#"*epoch + str(epoch)+ '\t'+str(loss.item()),end = '')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    writer.add_scalar('loss-epoch',loss.item(),epoch)

def test(epoch):
    total = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            x_data, y_data = data
            pred = model(x_data)
            _,predicted = torch.max(pred.data,dim=1)
            total+=x_data.size(0)
            correct+= (predicted==y_data).sum().item()
        print('\r'+"correct rate ="+str(correct/total),end='')
        time.sleep(1)
    writer.add_scalar('correct rate - epoch', correct/total, epoch)
start = time.time()
for epoch in range(100):
    train(epoch)
    test(epoch)
end = time.time()

# Produced By Bear. 22nd, Dec, 2021.
```
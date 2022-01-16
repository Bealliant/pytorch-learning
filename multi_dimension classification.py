import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import time

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

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307),(0.3081))
])

train_dataset = datasets.MNIST(root="/cifar-10-batches-py",train=True,transform=transforms,download=True)
train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
test_dataset = datasets.MNIST(root="/cifar-10-batches-py",train=False, transform= transforms)
test_loader = datasets.MNIST(root="cifar-10-batches-py",train=False,transform=transforms,download=True)

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




dummy_input = torch.rand(20, 1, 28, 28)  # 假设输入20张1*28*28的图片
with SummaryWriter(comment='LeNet') as w:
    w.add_graph(model, (dummy_input,))


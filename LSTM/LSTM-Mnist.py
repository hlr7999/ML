import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCH = 3
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
HIDDEN_SIZE = 64
LR = 0.01
DOWNLOAD_MNIST = False

train_data = dsets.MNIST(
    root = './mnist',
    train = True,
    transform = torchvision.transforms.ToTensor(),
    download = DOWNLOAD_MNIST,
)

test_data = torchvision.datasets.MNIST(root='./mnist',train=False)

train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)

test_x = Variable(torch.unsqueeze(test_data.test_data,dim=1),volatile=True).type(torch.FloatTensor)/255
test_y = test_data.test_labels

class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
 
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
        )
 
        self.out = nn.Linear(HIDDEN_SIZE,10)
 
    def forward(self,x):
        r_out, (h_n, h_c) = self.rnn(x, None)
 
        out = self.out(r_out[:,-1,:])
        return out

rnn = RNN().to(device)
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(),lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step,(x,y) in enumerate(train_loader):
        b_x = Variable(x.view(-1,28,28).to(device))
        b_y = Variable(y.to(device))
 
        output = rnn(b_x)
        loss = loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        if step%50 == 0:
            test_output = rnn(test_x.view(-1,28,28).to(device))
            pred_y = torch.max(test_output,1)[1].data.cpu().squeeze()
            accuracy = sum(pred_y == test_y)/float(test_y.size(0))
            print('Epoch: ',epoch, '| train loss:%.4f' %loss.data.item(),'| test accuracy:%.2f' %accuracy)

test_output = rnn(test_x[:10].view(-1,28,28).to(device))
pred_y = torch.max(test_output,1)[1].data.cpu().numpy().squeeze()
print(pred_y,'prediction number')
print(test_y[:10].numpy(),'real number')

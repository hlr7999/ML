import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torch.utils.data as Data
import torchvision

from lstm import LSTM

EPOCH = 3
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
HIDDEN_SIZE = 64
NUM_LAYERS = 1
LR = 0.01
DOWNLOAD_MNIST = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = dsets.MNIST(
    root = './mnist',
    train = True,
    transform = torchvision.transforms.ToTensor(),
    download = DOWNLOAD_MNIST,
)
test_data = dsets.MNIST(root='./mnist', train=False)

train_loader = Data.DataLoader(dataset=train_data, \
    batch_size=BATCH_SIZE, shuffle=True)

with torch.no_grad():
    test_x = Variable(torch.unsqueeze(test_data.data, dim=1)).type(torch.FloatTensor)/255
test_y = test_data.targets

lstm = LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, 10).to(device)

optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x.view(-1, TIME_STEP, INPUT_SIZE).to(device))
        b_y = Variable(y.to(device))
 
        output = lstm(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        if step%50 == 0:
            test_output = lstm(test_x.view(-1, TIME_STEP, INPUT_SIZE).to(device))
            pred_y = torch.max(test_output, 1)[1].data.cpu().squeeze()
            accuracy = sum(pred_y == test_y)/float(test_y.size(0))
            print('Epoch: ',epoch, '| train loss:%.4f' %loss.data.item(),'| test accuracy:%.2f' %accuracy)

test_output = lstm(test_x[:10].view(-1, TIME_STEP, INPUT_SIZE).to(device))
pred_y = torch.max(test_output,1)[1].data.cpu().numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')

torch.save(lstm.state_dict(), './model/model.ckpt')

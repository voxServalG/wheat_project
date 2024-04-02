from model.mobilenetv2 import *
from datasets import MyDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os
from pathlib import Path



batch_size = 32
shuffle = True
num_workers = 4
num_epochs = 100
max_epochs = 400
device = 'cuda' if torch.cuda.is_available() else 'cpu'
test_interval = 1
save_interval = 10
start_epoch = -1

use_L1_regularization = False
L1_lambda = 1e-2


writer = SummaryWriter()

model = MobileNetV2().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999))
mytransform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456,0.406), std=(0.229, 0.224, 0.225))
])

train_dataset = MyDataset(path_dir='./datasets/train',
                    transform=mytransform)
test_dataset = MyDataset(path_dir="./datasets/test",
                        transform=mytransform)

train_dataloader = DataLoader(dataset=train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers)
test_dataloader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)

def train_mobilenet(model, dataloader):
    model.train()
    running_loss = 0.0  
    for i, data in enumerate(dataloader, 0):
        print("START TRAINING: BATCH {}".format(i + 1))
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.to(torch.float32)
        # 梯度清零  
        model.zero_grad() 

        # 前向传播 + 反向传播 + 优化  
        outputs = model(inputs)  
        outputs = outputs.to(torch.float32)
        loss = criterion(outputs, labels)
        
        if use_L1_regularization:
            L1_reg = 0  
            for param in model.parameters():  
                L1_reg += torch.norm(param, 1)
            loss += L1_lambda * L1_reg
            
        running_loss += loss.item()
        loss.backward()  
        optimizer.step()  


        # 打印统计信息   
        print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(  
            i * batch_size,
            len(dataloader.dataset),  
            100. * i / len(dataloader),
            loss.item()))
    return running_loss / len(dataloader) 


def test_mobilenet(model, dataloader):
    model.eval()
    loss = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
    return loss.item()


def run_mobilenet(model, train_dataloader, test_dataloader, num_epochs):
    for epoch in range(start_epoch+1, max_epochs):
        print("START TRAINING: EPOCH {}".format(epoch + 1))
        train_loss = train_mobilenet(model=model, dataloader=train_dataloader)
        writer.add_scalar('Loss/train', train_loss, (epoch + 1) * len(train_dataloader))
        print(f'Epoch {epoch + 1}, Train loss: {train_loss:.6f}')

        if (epoch + 1) % test_interval == 0:
            test_loss = test_mobilenet(model=model, dataloader=test_dataloader)
            writer.add_scalar('Loss/test', test_loss, epoch + 1)  
            print(f'Epoch {epoch + 1}, Val loss: {test_loss:.6f}')

        if (epoch + 1) % save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, f'saved_weights/mobilenet/model_epoch_{epoch + 1}.pth')
            print("SAVED! epoch {}".format(epoch + 1))
    writer.close()
    print('Finished Training')

if __name__ == '__main__':
    Path("/saved_weights/mobilenet").mkdir(parents=True, exist_ok=True)
    '''
    pathlib的mkdir接收两个参数：
    parents：如果父目录不存在，是否创建父目录。
    exist_ok：只有在目录不存在时创建目录，目录已存在时不会抛出异常。
    '''
    run_mobilenet(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader, num_epochs=num_epochs)

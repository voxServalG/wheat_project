from model.edgevit import *
from datasets import MyDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

batch_size = 8
shuffle = True
num_workers = 2
num_epochs = 10

model = EdgeViT_S()
criterion = nn.MSELoss()
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.9,0.999))

mytransform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = MyDataset(path_dir='./datasets',
                    transform=mytransform)

dataloader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers)


def train():
    for epoch in range(num_epochs):  
        running_loss = 0.0  
        for i, data in enumerate(dataloader, 0):  
            inputs, labels = data  
    
            # 梯度清零  
            optimizer.zero_grad()  
    
            # 前向传播 + 反向传播 + 优化  
            outputs = model(inputs)  
            loss = criterion(outputs, labels)  
            loss.backward()  
            optimizer.step()  
    
            # 打印统计信息  
            running_loss += loss.item()  
            if i % 2000 == 1999:  # 每2000个mini-batches打印一次  
                print('[%d, %5d] loss: %.3f' %  
                    (epoch + 1, i + 1, running_loss / 2000))  
                running_loss = 0.0  
    
    print('Finished Training')


if __name__ == '__main__':
    train()
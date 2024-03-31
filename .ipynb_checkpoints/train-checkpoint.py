from model.edgevit import *
from datasets import MyDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from tensorboardX import SummaryWriter

batch_size = 8
shuffle = True
num_workers = 2
num_epochs = 100
max_epochs = 1000000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
test_interval = 5
save_interval = 5
start_epoch = -1

writer = SummaryWriter()

model = EdgeViT_S().to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.999))
# optimizer = optim.RMSprop(model.parameters(), lr=1e-6, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0)
# optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.5)
mytransform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])



# ============================ step 5+/5 断点恢复 ============================
load_checkpoint = 0
if load_checkpoint:
    print("LOADING CHECKPOINT")
    path_checkpoint = "./saved_weights/edgevits/model_epoch_100.pth"
    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print("LOAD SUCCESS")





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


def test_edgevit(model, dataloader):
    model.eval()
    loss = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
    return loss.item()

# Train for an epoch and return average batch loss as LOSS OF CURRENT EPOCH
def train_edgevit(model, dataloader):
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
        running_loss += loss.item()
        loss.backward()  
        optimizer.step()  


        # 打印统计信息   
        print('[{}/{} ({:.0f}%)]\tLoss: {:.2f}'.format(  
            i * batch_size,
            len(dataloader.dataset),  
            100. * i / len(dataloader),
            loss.item()))
    return running_loss / len(dataloader)   
        
    
def run_edgevit(model, train_dataloader, test_dataloader, num_epochs):
    for epoch in range(start_epoch+1, max_epochs):
        print("START TRAINING: EPOCH {}".format(epoch + 1))
        train_loss = train_edgevit(model=model, dataloader=train_dataloader)
        writer.add_scalar('Loss/train', train_loss, (epoch + 1) * len(train_dataloader))
        print(f'Epoch {epoch + 1}, Train loss: {train_loss:.2f}')

        if (epoch + 1) % test_interval == 0:
            test_loss = test_edgevit(model=model, dataloader=test_dataloader)
            writer.add_scalar('Loss/test', test_loss, epoch + 1)  
            print(f'Epoch {epoch + 1}, Val loss: {test_loss:.2f}')

        if (epoch + 1) % save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, f'saved_weights/edgevits/model_epoch_{epoch + 1}.pth')
            print("SAVED! epoch {}".format(epoch + 1))
    writer.close()
    print('Finished Training')


if __name__ == '__main__':
    run_edgevit(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader, num_epochs=num_epochs)

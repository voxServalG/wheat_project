import torch
from torchvision import transforms
from PIL import Image
from model.edgevit import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 定义加载模型的函数  
def load_model(model_path):  
    # 加载模型  
    model = EdgeViT_S().to(device)  # 替换为你的模型类
    model.load_state_dict(torch.load(model_path,map_location=torch.device(device)))  # 加载模型权重  
    model.eval()  # 设置模型为评估模式  
    return model 

# 定义推理函数  
def inference(model, image_path):  
    # 图像预处理  
    mytransform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')  
    input_tensor = preprocess(image)  
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model  
  
    # 推理  
    with torch.no_grad():  
        output = model(input_batch)  
    _, predicted_idx = torch.max(output, 1)  
    return predicted_idx.item()  
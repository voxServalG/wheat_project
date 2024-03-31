import torch
from torchvision import transforms
from PIL import Image
from model.edgevit import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 定义加载模型的函数  
def load_model(model_path):  
    # 加载模型  
    model = EdgeViT_S().to(device)  # 替换为你的模型类
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))  # 加载模型权重  
    model.eval()  # 设置模型为评估模式  
    return model 
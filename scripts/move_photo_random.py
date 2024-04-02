import os  
import random  
import shutil  
  
# 原始文件夹路径  
source_folder = '/home/vox/Documents/codes/python/wheat_project/datasets/train/photos'  
# 目标文件夹路径，如果不存在，代码会尝试创建它  
target_folder = '/home/vox/Documents/codes/python/wheat_project/datasets/val/photos'  
  
# # 确保目标文件夹存在  
# if not os.path.exists(target_folder):  
#     os.makedirs(target_folder)  
  
# 获取源文件夹中所有文件的列表  
files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]  
  
# 检查文件数量是否足够  
if len(files) < 95:  
    print("源文件夹中的文件数量不足95个，无法执行操作。")  
else:  
    # 随机选择95个文件  
    selected_files = random.sample(files, 95)  
      
    # 将选中的文件移动到目标文件夹  
    for file in selected_files:  
        source_path = os.path.join(source_folder, file)  
        target_path = os.path.join(target_folder, file)  
        shutil.move(source_path, target_path)  
        print(f"文件 {file} 已移动到目标文件夹。")  
  
print("操作完成。")
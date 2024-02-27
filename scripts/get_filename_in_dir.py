import os  
import pandas as pd  
  
# 指定文件夹路径  
folder_path_list = os.listdir("../photos")
print(folder_path_list) 

# # 获取文件夹中的文件名列表  
# file_names = os.listdir(folder_path)  
  
# # 创建一个DataFrame来存储文件名  
# df = pd.DataFrame(file_names, columns=['文件名'])  
  
# # 将DataFrame保存到表格文件中  
# df.to_csv('文件名列表.csv', index=False, encoding='utf-8-sig')  
  
# print('文件名已列到表格上，并保存到文件名列表.csv文件中。')
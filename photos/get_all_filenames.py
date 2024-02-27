import os  
import pandas as pd  
  
# 指定文件夹路径  
folder_dir_list = ['20230304', '20221105', '20230225', '20221109', '20221029', '20221116-1124', '20221102']

for folder_dir in folder_dir_list:
    # 获取文件夹中的文件名列表  
    file_names = os.listdir(folder_dir)  
    file_names = sorted(file_names)
    # 创建一个DataFrame来存储文件名  
    df = pd.DataFrame(file_names, columns=['filename'])  
    
    # 将DataFrame保存到表格文件中
    csv_path = "{}/summary.csv".format(folder_dir)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')  
    
    print('文件名已列到表格上，并保存到文件名列表.csv文件中。')

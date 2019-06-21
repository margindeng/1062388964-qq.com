import os
#设定文件路径
path='talk.politics.misc/'
i=1

#对目录下的文件进行遍历
for file in os.listdir(path):
    # 判断是否是文件
    if os.path.isfile(os.path.join(path,file))==True:
        new_name=file.replace(file,"%d.txt"%i)#设置新文件名
        os.rename(os.path.join(path,file),os.path.join(path,new_name))#重命
        i+=1


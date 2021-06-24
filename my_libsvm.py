from libsvm.svmutil import *
from liblinear.liblinearutil import *
import cv2
import os
import numpy as np
import fnmatch
import random
from datetime import datetime
seed = 1
random.seed(seed)
np.random.seed(seed)

# 从 0~n 中随机选取 x 个数字
def getRandomIdx(n, x):
    return np.random.choice(np.arange(n), size=x, replace=False)

# 批量获取并处理图片
list = []
numlist = [] #记录每个类别有效的图片数
path=r'C:\Users\jincheng\PR\yaleBExtData\yaleB'
for i in [x for x in range(1,40) if x != 14]:
    k = str(i).zfill(2)
    file_path = path + k
    files = os.listdir(file_path)
    count=0
    for file in files:
        if fnmatch.fnmatch(str(file),'*P00A*') and not fnmatch.fnmatch(str(file),'*bad') :
            count+=1
            img_src = cv2.imread(os.path.join(file_path,file),cv2.IMREAD_GRAYSCALE)
            img_result = cv2.pyrDown(img_src)
            img_result = cv2.pyrDown(img_result)
            img_result = img_result.reshape(2016)/255 #归一化
            img_result=np.insert(img_result,0,int(file_path[-2:]))
            list.append(img_result)
    numlist.append(count)

with open('data.txt','w') as f:    #将所有用到的数据写入一个文件data.txt
    for l in list:
        line = str(int(l[0]))
        for i in range (1,len(l)):
            line += " %d:%.3f " % (i,l[i])
        line+='\n'
        f.write(line)
y,x=svm_read_problem('data.txt')
# 划分训练集与测试集
for p in [7,13,20]:
    print('p=',p)
    train_index = getRandomIdx(numlist[0]-1,p)
    all_index = [x for x in range(len(y))]
    for i in range(1,38):
        train_index = np.append(train_index,getRandomIdx(numlist[0]-1,p) + sum(numlist[0:i]))
    test_index = np.delete(all_index,train_index)
    x_train = np.array(x)[train_index].tolist()
    y_train = np.array(y)[train_index].tolist()
    x_test = np.array(x)[test_index].tolist()
    y_test = np.array(y)[test_index].tolist()

    # 将训练用的数据集写入data_train_p.txt文件
    name='data_train_' + str(p) + '.txt'
    print(name,"is generated successfully.")
    with open(name,'w') as f:    #设置文件对象
        for l in np.array(list)[train_index].tolist():
            line = str(int(l[0]))
            for i in range (1,len(l)):
                line += " %d:%.3f " % (i,l[i])
            line+='\n'
            f.write(line)
    f.close()

# 对不同的p分别使用5-fold 选择最佳参数，
print("Using 5-fold CV in the training set to get the parameter C.")
bestC_list = [] 
best_acc = -1
for p in [7,13,20]:
    name='data_train_' + str(p) + '.txt'
    y_train,x_train=svm_read_problem(name)
    prob = svm_problem(y_train, x_train)
    C_list = [2**i for i in range(-4,16,4)]
    print("\n p=",p,"selecting C from",str(C_list))
    
    
    for i in range(len(C_list)):
        cmd = '-t 0 -v 5 -c '+ str(C_list[0])
        param = svm_parameter(cmd)
        acc = svm_train(prob,param)
        if acc > best_acc:
            best_acc = acc
            bestC = C_list[i]
            print("when p is",p,"the bestC is",bestC,",the best accuracy is ",best_acc)
    bestC_list.append(bestC)
print("\nSo, the bestC_list when p=7,13,20 is ",str(bestC_list),"separately")


# 根据最佳参数训练出三个模型并测试
k=0
num=10
time_cost = np.zeros([3,num])##时间记录
test_accuracy = np.zeros([3,num])##准确度记录
for p in [7,13,20]:
    for j in range(num): #测试num次
        train_index = getRandomIdx(numlist[0]-1,p)
        for i in range(1,38):
            train_index = np.append(train_index,getRandomIdx(numlist[0]-1,p) + sum(numlist[0:i]))
        test_index = np.delete(all_index,train_index)
        x_train = np.array(x)[train_index].tolist()
        y_train = np.array(y)[train_index].tolist()
        x_test = np.array(x)[test_index].tolist()
        y_test = np.array(y)[test_index].tolist()
        prob_train = svm_problem(y_train, x_train)
        prob_test = svm_problem(y_test, x_test)
        cmd = '-t 0 -c '+str(bestC_list[k])
        model = svm_train(prob_train,cmd) #在训练集上用最优参数训练模型
        t0 = datetime.now()
        _, p_acc, _ = svm_predict(y_test, x_test, model) ##在测试集上测试模型，并计算时间和准确度
        t1 = datetime.now()
        time = (t1-t0).total_seconds() / len(y_test)
        test_accuracy[k,j] = p_acc[0]
        time_cost[k,j] = time
    print("When p =",p,"the accuracy of classification in the test set is",test_accuracy.mean(1)[k],"%")
    print("When p =",p,"the time cost per picture is",time_cost.mean(1)[k],"seconds.")
    k+=1
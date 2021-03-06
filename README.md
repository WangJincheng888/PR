# PR
模式识别大作业

Project I
基于支持向量机和稀疏表示分类的人脸识别比较研究

1.实验数据
   数据文件夹“ftp://public.sjtu.edu.cn/imageData/faceImage/yaleBExtData”中包含38个人的人脸图片，每人64幅共计2432幅大小为192×168的图片（去掉每个文件夹中的第一个图像文件，即含“Ambient”的文件），将每幅图降采样为48×42大小的图像后拉伸为一个一维的向量。随机地从每个人的图片中取p幅图片作为训练样本（p取7、13或20），38个人总共38p个训练样本，剩下的图片作为测试样本。

2.实验内容
    分别采用线性SVM分类器和稀疏表示分类方法进行人脸分类识别，要求:
1) 线性SVM分类器采用两种形式：LIBSVM和LIBLINEAR；
2) 稀疏表示分类采用两种字典：(1)全部训练样本（SRC）；(2)DKSVD字典；
3）只能用训练样本训练分类器或优化字典；
4）只能用测试样本统计识别的精度；
5）对实验中的参数采用5-fold交叉验证进行选择。
分别对p=7、13、20进行实验（DKSVD字典的大小分别为200、400、600），每组实验重复10次，给出测试集上人脸识别的平均精度和识别每幅图片所用的平均时间(以秒为单位)。

3. 实验报告的主要内容:
   1) 实验目的及采用的方法；
   2) 采用方法的描述及实现算法；
   3) 实验数据及实验参数选择；
   4) 实验结果及对实验结果的分析和结论；
   5) 算法实现的主要MATLAB或Python代码。。

clc,clear
rng(1);
num=1;%重复10次实验
time_cost = zeros([3,num]);%时间记录
test_accuracy =zeros([3,num]);%准确度记录
p_list=[7,13,20];
for f = 1:3
    p=p_list(f);%7,13,20
    for n =1:10
    %读取人脸数据、降采样，形成数据集和标签集
        outerdir = 'C:\Users\jincheng\PR\yaleBExtData';
        files = dir(outerdir);%dir 列出指定路径下的所有子文件夹和子文件
        label = [];
        allclass = [];
        for i = 1:38
          K=i+2;
          foldername = files(K).name;
          currentdir = strcat(outerdir, '/', foldername);
          images = dir([currentdir '/*P00A*.pgm']);
          oneclass = [];
          for j = 1:length(images)-1
              img = imread([images(j).folder,'\',images(j).name]);
              img = double(img(1:4:end, 1:4:end));   
              feature = reshape(img,2016,1);  
              oneclass = [oneclass,feature];  
              allclass = [allclass,feature]; 
              label = [label,i];   
          end 
           Allsample{i} = oneclass;   
        end
        x_train = [];
        x_test = [];
        y_train = [];
        y_test = [];
        for i = 1:length(Allsample)
            m = size(Allsample{i},2);
            randse = randperm(m);
            train_one = Allsample{i}(:,randse(1:p));
            test_one = Allsample{i}(:,randse(p+1:m));
            trainlabel_one = i * ones(1,p);
            testlabel_one = i * ones(1,m-p);
            x_train = [x_train,train_one];
            x_test = [x_test,test_one];
            y_train = [y_train,trainlabel_one];
            y_test = [y_test,testlabel_one];
        end
        trainNorm = x_train./255;
        testNorm = x_test./255;

        testNum = size(testNorm,2);   
        trainNum = size(trainNorm,2);    
        labelpre = zeros(1,testNum);    
        classnum = length(Allsample);
         h = waitbar(0,'please wait...');   %进度条函数，数值在0~1

         for i = 1:testNum
            t1 = clock;
            xp = SolveHomotopy_CBM_std(trainNorm,testNorm(:,i),'lambda',0.01);
            r = zeros(1, classnum);
            for j = 1:classnum
                xn = zeros(trainNum,1);                   
                index = (j==y_train);   
                xn(index) = xp(index);   
                r(j) = norm((testNorm(:,i) - trainNorm * xn));  
            end

            [~,p] = min(r);    
            labelpre(i) = p;
            t2 = clock;
            ttime(i) = etime(t2,t1);
            percent = i / testNum;
            waitbar(percent,h,sprintf('p=%d的第%d次实验：%2.0f%%',p,n,percent*100));   
         end

        close(h);
        time = mean(ttime,2);
        accuracy = sum(labelpre == y_test) / testNum;

        disp(["第"+n+'次实验，p='+p+'，准确率为'+accuracy*100+'%'])
        disp(["第"+n+'次实验，p='+p+'，平均每幅图片消耗时间为'+time+'s'])
        time_cost(f,n) = time;    
        test_accuracy(f,n) = accuracy;
    end 
end




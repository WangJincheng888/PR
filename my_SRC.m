clc,clear
rng(1);
num=1;%�ظ�10��ʵ��
time_cost = zeros([3,num]);%ʱ���¼
test_accuracy =zeros([3,num]);%׼ȷ�ȼ�¼
p_list=[7,13,20];
for f = 1:3
    p=p_list(f);%7,13,20
    for n =1:10
    %��ȡ�������ݡ����������γ����ݼ��ͱ�ǩ��
        outerdir = 'C:\Users\jincheng\PR\yaleBExtData';
        files = dir(outerdir);%dir �г�ָ��·���µ��������ļ��к����ļ�
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
         h = waitbar(0,'please wait...');   %��������������ֵ��0~1

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
            waitbar(percent,h,sprintf('p=%d�ĵ�%d��ʵ�飺%2.0f%%',p,n,percent*100));   
         end

        close(h);
        time = mean(ttime,2);
        accuracy = sum(labelpre == y_test) / testNum;

        disp(["��"+n+'��ʵ�飬p='+p+'��׼ȷ��Ϊ'+accuracy*100+'%'])
        disp(["��"+n+'��ʵ�飬p='+p+'��ƽ��ÿ��ͼƬ����ʱ��Ϊ'+time+'s'])
        time_cost(f,n) = time;    
        test_accuracy(f,n) = accuracy;
    end 
end




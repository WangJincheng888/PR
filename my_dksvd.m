clc,clear
rng(1);
num=10;%重复10次实验
time_cost = zeros([3,num]);%时间记录
test_accuracy =zeros([3,num]);%准确度记录
p_list=[7,13,20];
dictsize_list=[200,400,600];
h = waitbar(0,'please wait...');   %进度条函数，数值在0~1
for f = 1:3
    p=p_list(f);%7,13,20
    dictsize = dictsize_list(f);%200,400,600
    outerdir = 'C:\Users\jincheng\PR\Face-Recognition\yaleBExtData';
    files = dir(outerdir);
    
    numList=[]; %记录每一类的有效图片个数
    for i = 1:38
        K=i+2;
        foldername = files(K).name;
        currentdir = strcat(outerdir, '/', foldername);
        images = dir([currentdir '/*P00A*.pgm']);
        numList = [numList,size(images, 1)];
    end
    for n = 1:10
        trainfeaturesArr = [];
        testfeaturesArr = [];
        %分别构建数据集和训练集的一维向量
        all_index = 1:sum(numList);
        train_index=[];
        for i = 1:38
            waitbar(i/38,h,sprintf('p=%d, 字典大小为%d, 第%d次实验：%2.0f%%',p,dictsize,n,i/38*100));   %显示每个样本测试进度条  
            randomarray = randperm(numList(i));
            train_index = [train_index, randomarray(1:p) + sum(numList(1:i-1))];
            K=i+2;
            foldername = files(K).name;
            currentdir = strcat(outerdir, '/', foldername);
            images = dir([currentdir '/*P00A*.pgm']);
            for j= 1:length(images)
                img = imread(strcat(currentdir, '/', images(j).name));
                img = double(img(1:4:end,1:4:end));
                feature = reshape(img,2016,1)/255;
                if ismember(j, randomarray(1:p))
                    trainfeaturesArr = horzcat(trainfeaturesArr,feature);
                else
                    testfeaturesArr = horzcat(testfeaturesArr,feature);
                end
            end
        end
        % test_index = setdiff(all_index,train_index);
        %真值表
        H_train = zeros(38,p*38);
        H_test = zeros(38,sum(numList)-p*38);
        H_train(1,1:p)=1;
        H_test(1,1:numList(1)-p)=1;
        for i = 2:38
            H_train(i,((i-1)*p+1):(i*p))=1;
            test_num = numList(i)-p;
            H_test(i,(sum(numList(1:i-1))-p*(i-1)+1 ):(sum(numList(1:i-1))-p*(i-1) + test_num) )=1;
        end

        sqrt_gamma = 2;
        params.data = [trainfeaturesArr; sqrt_gamma * H_train];
        params.Tdata = 30; % T0
        params.dictsize = dictsize; % 200,400,600
        params.iterations = 100;
        [ksvd_dict, ksvd_Gamma, ksvd_err] = ksvd(params,'');

        dksvd_dict = ksvd_dict(1: size(trainfeaturesArr, 1) , :);
        dksvd_w = ksvd_dict(size(trainfeaturesArr, 1) +1: size(ksvd_dict, 1), :);

        l2norms = sqrt(sum(dksvd_dict.*dksvd_dict,1)+eps);
        D = dksvd_dict ./ repmat(l2norms,size(dksvd_dict,1),1);
        dksvd_w_norm = dksvd_w ./ repmat(l2norms,size(dksvd_w,1),1);
        W = dksvd_w_norm ./ sqrt_gamma;

        sparsity = 30;
        gamma  = omp(D'*testfeaturesArr, D'*D, sparsity);
        true_class = [];
        for j = 1: size(H_test, 2)
            [~, I1] = max(H_test(:, j));
            true_class(j) = I1;
        end
        t1 = clock;
        pred = [];
        for i = 1: size(gamma, 2)
            l = W * gamma(:, i);
            [~, I] = max(l);
            pred(i) = I;
        end
        t2 = clock;
        time = etime(t2,t1)/ (size(pred, 2));
        accuracy = ((sum(pred == true_class)) * 100)/ (size(pred, 2));
        
        test_accuracy(f,n) = accuracy;
        time_cost(f,n) = time;
        disp(["第"+n+'次实验，p='+p+'，字典大小为'+dictsize+'，准确率为'+accuracy+'%'])
        disp(["第"+n+'次实验，p='+p+'，字典大小为'+dictsize+'，平均每幅图片消耗时间为'+time+'s'])
    end
end







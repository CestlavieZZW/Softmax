function [traindata,testdata,trainlab,testlab] = cross1_10(X, y, k, ratio)

[N,M] = size(X);
numtemp = 1/(1-ratio);
num = floor(numtemp);

y_labels = unique(y);

traindata = zeros(N*ratio,M,num);
trainlab = zeros(N*ratio,num);
testdata = zeros((N*round(1-ratio)),M,num);
testlab  = zeros((N*round(1-ratio)),num);
xdatatemp = zeros(N/k,M);

for j = 1:k
    comm_i = find(y == y_labels(j));
    % Data = rand(9,3);%创建维度为9×3的随机矩阵样本
    indices = crossvalind('Kfold', 200, num);%将数据样本随机分割为10部分
  
    for t = 1:length(comm_i)
        xdatatemp(t,:) = X(comm_i(t),:);
    end
    
    for i = 1:10 %循环10次，分别取出第i部分作为测试样本，其余9部分作为训练样本
        test = (indices == i);
        train = ~test;
        traindata((180*(j-1)+1):180*(j),:,i)=xdatatemp(train, :);
        trainlab((180*(j-1)+1):180*(j),i)=j;
        testdata((20*(j-1)+1):20*(j),:,i)=xdatatemp(test, :);
        testlab((20*(j-1)+1):20*(j),i)=j;
    end    
end
end

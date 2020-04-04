function [rate1,rate2] = ave(testdata,traindata,trainlab,testlab,lambda,alpha,MAX_ITR)

y_labels = unique(trainlab); 
ydiff = length(y_labels);
[~,xsize,~] = size(traindata);

theta = zeros(xsize,ydiff,10);
rate = zeros(10,1);
rate_test = zeros(10,1);

for i = 1:10
    [theta(:,:,i),test_pre,rate(i)] = mysoftmax_gd(testdata(:,:,i),traindata(:,:,i),trainlab(:,i),lambda,alpha,MAX_ITR);
    index_t = find(testlab(:,i)==test_pre); % �ҳ�Ԥ����ȷ��������λ��
    rate_test(i) = length(index_t)/length(testlab(:,i)); % ����Ԥ�⾫��
end
rate1 = mean(rate);
rate2 = mean(rate_test);
end
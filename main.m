clear;
clc;

Sigma = [1, 0; 0, 1];
mu1 = [1, -1];
x1 = mvnrnd(mu1, Sigma, 200);
y1 = ones(200,1);
mu2 = [5, -4];
x2 = mvnrnd(mu2, Sigma, 200);
y2 = 2*ones(200,1);
mu3 = [1, 4];
x3 = mvnrnd(mu3, Sigma, 200);
y3 = 3*ones(200,1);
mu4 = [6, 4.5];
x4 = mvnrnd(mu4, Sigma, 200);
y4 = 4*ones(200,1);
mu5 = [7.5, 0.0];
x5 = mvnrnd(mu5, Sigma, 200);
y5 = 5*ones(200,1);
bias = ones(1000,1);

% Show the data points
figure
plot(x1(:,1), x1(:,2), 'r.'); 
hold on;
plot(x2(:,1), x2(:,2), 'b.');
hold on;
plot(x3(:,1), x3(:,2), 'k.');
hold on;
plot(x4(:,1), x4(:,2), 'g.');
hold on;
plot(x5(:,1), x5(:,2), 'm.');

xdata = [x1; x2; x3; x4; x5];
xdatafinal = [bias,bias,bias,xdata];
ylab = [y1; y2; y3; y4; y5];
[xtrain,ytrain,xtest,ytest] = divide(xdatafinal,ylab,5,0.6);

lambda = 0.001; % Ȩ��˥������Weight decay parameter
alpha = 0.01; % ѧϰ����
MAX_ITR=500; % ����������
N = 50;
rate = zeros(N,1);
rate_test = zeros(N,1);
i = 1;

tic
while i<=50
[theta,test_pre,rate(i)] = mysoftmax_gd(xtest,xtrain,ytrain,lambda,alpha,MAX_ITR);
index_t = find(ytest==test_pre); % �ҳ�Ԥ����ȷ��������λ��
rate_test(i) = length(index_t)/length(ytest); % ����Ԥ�⾫��
    lambda = lambda + 0.1;
    i = i+1;
end
toc

lmd = 0.001:0.1:4.901;
figure
plot(lmd,rate);
hold on
plot(lmd,rate_test);
xlabel('lambda')
ylabel('Accuracy')
legend('train','test');

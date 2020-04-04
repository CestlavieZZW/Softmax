clear;
clc;

lambda = 100; % 权重衰减参数Weight decay parameter
alpha = 0.01; % 学习速率
MAX_ITR=500; % 最大迭代次数

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
ratio = 0.9;
k = 5;

[traindata,testdata,trainlab,testlab] = cross1_10(xdatafinal, ylab, k, ratio);
[r1,r2] = ave(testdata,traindata,trainlab,testlab,lambda,alpha,MAX_ITR);



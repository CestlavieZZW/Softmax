function [theta,test_pre,rate] = mysoftmax_gd(X_test,X,label,lambda,alpha,MAX_ITR)
% 该函数用于实现梯度下降法softmax回归
% 调用方式：[theta,test_pre,rate] = mysoftmax_gd(X_test,X,label,lambda,alpha,MAX_ITR,varargin)
% X_test：测试输入数据
% X：训练输入数据，组织为m*p矩阵，m为案例个数，p为加上常数项之后的属性个数
% label：训练数据标签，组织为m*1向量（数值型）
% lambda：权重衰减参数weight decay parameter
% alpha：梯度下降学习速率
% MAX_ITR：最大迭代次数
% theta：梯度下降法的theta系数寻优结果
% test_pre：测试数据预测标签
% rate：训练数据回判正确率

% 梯度下降寻优

[m,p] = size(X);
numClasses = length(unique(label)); % 求取标签类别数
theta = 0.005*randn(p,numClasses); % 随机初始化系数
cost=zeros(MAX_ITR,1); % 用于追踪代价函数的值

for k=1:MAX_ITR
    [cost(k),grad] = softmax_cost_grad(X,label,lambda,theta); % 计算代价函数值和梯度
    theta=theta-alpha*grad; % 更新系数
end

% 回判预测
[~,~,Probit] = softmax_cost_grad(X,label,lambda,theta);
[~,label_pre] = max(Probit,[],2);
index = find(label==label_pre); % 找出预测正确的样本的位置
rate = length(index)/m; % 计算预测精度
% 绘制代价函数图
% figure('Name','代价函数值变化图');
% plot(0:MAX_ITR-1,cost)
% xlabel('迭代次数'); ylabel('代价函数值')
% title('代价函数值变化图');% 绘制代价函数值变化图
% 测试数据预测
[mt,~] = size(X_test);
Probit_t = zeros(mt,length(unique(label)));
for smpt = 1:mt
    Probit_t(smpt,:) = exp(X_test(smpt,:)*theta)/sum(exp(X_test(smpt,:)*theta));
end
[~,test_pre] = max(Probit_t,[],2);

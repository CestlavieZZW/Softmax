function [theta,test_pre,rate] = mysoftmax_gd(X_test,X,label,lambda,alpha,MAX_ITR)
% �ú�������ʵ���ݶ��½���softmax�ع�
% ���÷�ʽ��[theta,test_pre,rate] = mysoftmax_gd(X_test,X,label,lambda,alpha,MAX_ITR,varargin)
% X_test��������������
% X��ѵ���������ݣ���֯Ϊm*p����mΪ����������pΪ���ϳ�����֮������Ը���
% label��ѵ�����ݱ�ǩ����֯Ϊm*1��������ֵ�ͣ�
% lambda��Ȩ��˥������weight decay parameter
% alpha���ݶ��½�ѧϰ����
% MAX_ITR������������
% theta���ݶ��½�����thetaϵ��Ѱ�Ž��
% test_pre����������Ԥ���ǩ
% rate��ѵ�����ݻ�����ȷ��

% �ݶ��½�Ѱ��

[m,p] = size(X);
numClasses = length(unique(label)); % ��ȡ��ǩ�����
theta = 0.005*randn(p,numClasses); % �����ʼ��ϵ��
cost=zeros(MAX_ITR,1); % ����׷�ٴ��ۺ�����ֵ

for k=1:MAX_ITR
    [cost(k),grad] = softmax_cost_grad(X,label,lambda,theta); % ������ۺ���ֵ���ݶ�
    theta=theta-alpha*grad; % ����ϵ��
end

% ����Ԥ��
[~,~,Probit] = softmax_cost_grad(X,label,lambda,theta);
[~,label_pre] = max(Probit,[],2);
index = find(label==label_pre); % �ҳ�Ԥ����ȷ��������λ��
rate = length(index)/m; % ����Ԥ�⾫��
% ���ƴ��ۺ���ͼ
% figure('Name','���ۺ���ֵ�仯ͼ');
% plot(0:MAX_ITR-1,cost)
% xlabel('��������'); ylabel('���ۺ���ֵ')
% title('���ۺ���ֵ�仯ͼ');% ���ƴ��ۺ���ֵ�仯ͼ
% ��������Ԥ��
[mt,~] = size(X_test);
Probit_t = zeros(mt,length(unique(label)));
for smpt = 1:mt
    Probit_t(smpt,:) = exp(X_test(smpt,:)*theta)/sum(exp(X_test(smpt,:)*theta));
end
[~,test_pre] = max(Probit_t,[],2);

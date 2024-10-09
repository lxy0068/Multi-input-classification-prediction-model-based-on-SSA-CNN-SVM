%%  ��ջ�������
warning off             % �رձ�����Ϣ
close all               % �رտ�����ͼ��
clear                   % ��ձ���
clc                     % ���������
%%  ��ȡ����
res = xlsread('DGAԭʼ����.xlsx');

%%  ��������
num_class = length(unique(res(:, end)));  % �������Excel���һ�з����
num_dim = size(res, 2) - 1;               % ����ά��
num_res = size(res, 1);                   % ��������ÿһ�У���һ��������
num_size = 0.7;                           % ѵ����ռ���ݼ��ı���
res = res(randperm(num_res), :);          % �������ݼ�������������ʱ��ע�͸��У�
flag_conusion = 1;                        % ��־λΪ1���򿪻�������Ҫ��2018�汾�����ϣ�

%%  ���ñ����洢����
P_train = []; P_test = [];
T_train = []; T_test = [];

%%  �������ݼ�
for i = 1 : num_class
    mid_res = res((res(:, end) == i), :);           % ѭ��ȡ����ͬ��������
    mid_size = size(mid_res, 1);                    % �õ���ͬ�����������
    mid_tiran = round(num_size * mid_size);         % �õ�������ѵ����������

    P_train = [P_train; mid_res(1: mid_tiran, 1: end - 1)];       % ѵ��������
    T_train = [T_train; mid_res(1: mid_tiran, end)];              % ѵ�������

    P_test  = [P_test; mid_res(mid_tiran + 1: end, 1: end - 1)];  % ���Լ�����
    T_test  = [T_test; mid_res(mid_tiran + 1: end, end)];         % ���Լ����
end

%%  ����ת��
P_train = P_train'; P_test = P_test';
T_train = T_train'; T_test = T_test';
%%  �õ�ѵ�����Ͳ�����������
M = size(P_train, 2);
N = size(P_test , 2);
%% ���ݹ�һ��
[P_train, ps_input] = mapminmax(P_train,0,1);
P_test = mapminmax('apply',P_test,ps_input);

t_train =  categorical(T_train)';
t_test  =  categorical(T_test)';

%%  ���ݷ���
outdim = 1;                                  % ���һ��Ϊ���
f_ = size(res, 2) - 1;               % ����ά��                  % ��������ά��

%%  ����ƽ��
%   ������ƽ�̳�1ά����ֻ��һ�ִ���ʽ
%   Ҳ����ƽ�̳�2ά���ݣ��Լ�3ά���ݣ���Ҫ�޸Ķ�Ӧģ�ͽṹ
%   ����Ӧ��ʼ�պ���������ݽṹ����һ��
p_train =  double(reshape(P_train, f_,1 , 1, M));
p_test  =  double(reshape(P_test , f_,1 , 1, N));
% t_train =  double(t_train)';
% t_test  =  double(t_test )'
%%  �Ż��㷨��������
SearchAgents_no = 8;                   % ����
Max_iteration = 10;                    % ����������
dim = 3;                               % �Ż���������
lb = [1e-3,64 ,1e-5];                 % ����ȡֵ�½�(ѧϰ�ʣ�������������ϵ��)
ub = [5e-2, 512,1e-2];                 % ����ȡֵ�Ͻ�(ѧϰ�ʣ�������������ϵ��)

fitness = @(x)fical(x);

[Best_score,Best_pos,curve]=SSA(SearchAgents_no,Max_iteration,lb ,ub,dim,fitness)
Best_pos(1, 2) = round(Best_pos(1, 2));   
best_hd  = Best_pos(1, 2); % ��������
best_lr= Best_pos(1, 1);% ��ѳ�ʼѧϰ��
best_l2 = Best_pos(1, 3);% ���L2����ϵ��     
%%  ��������ṹ
layers = [
 imageInputLayer([f_, 1])     % ����� �������ݹ�ģ[f_, 1, 1]
 
 convolution2dLayer([3, 1], 32)  % ����˴�С 3*1 ����16������ͼ
 batchNormalizationLayer         % ����һ����
 reluLayer                       % Relu�����
 
 convolution2dLayer([3, 1], 32)  % ����˴�С 3*1 ����32������ͼ
 batchNormalizationLayer         % ����һ����
 reluLayer                       % Relu�����

 dropoutLayer(0.1)               % Dropout��
 fullyConnectedLayer(num_class)          % ȫ���Ӳ�
 softmaxLayer
 classificationLayer];               % �����

%%  ��������
options = trainingOptions('sgdm', ...     % SGDM �ݶ��½��㷨
    'MiniBatchSize', best_hd,...               % ����С,ÿ��ѵ����������30
    'MaxEpochs', 100,...                  % ���ѵ������ 500
    'InitialLearnRate', best_lr,...          % ��ʼѧϰ��Ϊ0.01
    'LearnRateSchedule', 'piecewise',...  % ѧϰ���½�
    'LearnRateDropFactor', 0.1,...        % ѧϰ���½����� 0.1
    'LearnRateDropPeriod', 80,...        % ����400��ѵ���� ѧϰ��Ϊ 0.01*0.1
    'Shuffle', 'every-epoch',...          % ÿ��ѵ���������ݼ�
    'L2Regularization', best_l2, ...         % L2���򻯲���
    'Plots', 'training-progress',...      % ��������
    'Verbose', false);

%%  ѵ��ģ��
net = trainNetwork(p_train, t_train, layers, options);
inputSize = net.Layers(1).InputSize;

%% ���ݴ���
train =  p_train;
test  = p_test;

%% ��ȡ��ά����
layer = 'fc';
P_train = activations(net,train,layer,'OutputAs','rows');
P_test  = activations(net,test,layer,'OutputAs','rows');

%% �������
T_train = t_train;
T_test  = t_test;

%% ����/ѵ��SVMģ��
T_train = string(T_train);
T_test = string(T_test);

T_train = double(T_train); P_train = double(P_train); 
T_test  = double(T_test) ; P_test = double(P_test); 

cmd = [' -t 2',' -c ',num2str(100),' -g ',num2str(0.01)];
model = svmtrain(T_train,P_train,cmd);

%% SVM�������
T_sim1 = svmpredict(T_train,P_train,model);
T_sim2 = svmpredict(T_test,P_test,model);
%% ����ͼ
figure
plot(curve, 'linewidth',1.5);
title('��Ӧ��')
xlabel('The number of iterations')
ylabel('Fitness')
%% ����׼ȷ��
accuracy1 = mean(T_sim1 == T_train) * 100;
accuracy2 = mean(T_sim2 == T_test) * 100;

%% ��ӡ
disp(['ѵ����׼ȷ�ʣ�' num2str(accuracy1) '%'] )
disp(['���Լ�׼ȷ�ʣ�' num2str(accuracy2) '%'] )



%% ��ʽת��
T_sim1 = double(T_sim1); 
T_sim2 = double(T_sim2); 

M = length(T_train);
N = length(T_test);

%%  ��ͼ
figure
plot(1:M,T_train,'r-*',1:M,T_sim1,'b-o','LineWidth',1)
legend('��ʵֵ','SSA-CNN-SVMԤ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string={'ѵ����Ԥ�����Ա�';['׼ȷ��=' num2str(accuracy1) '%']};
title(string)
grid

figure
plot(1:N,T_test,'r-*',1:N,T_sim2,'b-o','LineWidth',1)
legend('��ʵֵ','SSA-CNN-SVMԤ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string={'���Լ�Ԥ�����Ա�';['׼ȷ��=' num2str(accuracy2) '%']};
title(string)
grid

%%  ��������
figure
cm = confusionchart(T_train, T_sim1);
cm.Title = 'Confusion Matrix for Train Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

figure
cm = confusionchart(T_test, T_sim2);
cm.Title = 'Confusion Matrix for Test Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
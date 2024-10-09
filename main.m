%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行
%%  读取数据
res = xlsread('DGA原始数据.xlsx');

%%  分析数据
num_class = length(unique(res(:, end)));  % 类别数（Excel最后一列放类别）
num_dim = size(res, 2) - 1;               % 特征维度
num_res = size(res, 1);                   % 样本数（每一行，是一个样本）
num_size = 0.7;                           % 训练集占数据集的比例
res = res(randperm(num_res), :);          % 打乱数据集（不打乱数据时，注释该行）
flag_conusion = 1;                        % 标志位为1，打开混淆矩阵（要求2018版本及以上）

%%  设置变量存储数据
P_train = []; P_test = [];
T_train = []; T_test = [];

%%  划分数据集
for i = 1 : num_class
    mid_res = res((res(:, end) == i), :);           % 循环取出不同类别的样本
    mid_size = size(mid_res, 1);                    % 得到不同类别样本个数
    mid_tiran = round(num_size * mid_size);         % 得到该类别的训练样本个数

    P_train = [P_train; mid_res(1: mid_tiran, 1: end - 1)];       % 训练集输入
    T_train = [T_train; mid_res(1: mid_tiran, end)];              % 训练集输出

    P_test  = [P_test; mid_res(mid_tiran + 1: end, 1: end - 1)];  % 测试集输入
    T_test  = [T_test; mid_res(mid_tiran + 1: end, end)];         % 测试集输出
end

%%  数据转置
P_train = P_train'; P_test = P_test';
T_train = T_train'; T_test = T_test';
%%  得到训练集和测试样本个数
M = size(P_train, 2);
N = size(P_test , 2);
%% 数据归一化
[P_train, ps_input] = mapminmax(P_train,0,1);
P_test = mapminmax('apply',P_test,ps_input);

t_train =  categorical(T_train)';
t_test  =  categorical(T_test)';

%%  数据分析
outdim = 1;                                  % 最后一列为输出
f_ = size(res, 2) - 1;               % 特征维度                  % 输入特征维度

%%  数据平铺
%   将数据平铺成1维数据只是一种处理方式
%   也可以平铺成2维数据，以及3维数据，需要修改对应模型结构
%   但是应该始终和输入层数据结构保持一致
p_train =  double(reshape(P_train, f_,1 , 1, M));
p_test  =  double(reshape(P_test , f_,1 , 1, N));
% t_train =  double(t_train)';
% t_test  =  double(t_test )'
%%  优化算法参数设置
SearchAgents_no = 8;                   % 数量
Max_iteration = 10;                    % 最大迭代次数
dim = 3;                               % 优化参数个数
lb = [1e-3,64 ,1e-5];                 % 参数取值下界(学习率，批量处理，正则化系数)
ub = [5e-2, 512,1e-2];                 % 参数取值上界(学习率，批量处理，正则化系数)

fitness = @(x)fical(x);

[Best_score,Best_pos,curve]=SSA(SearchAgents_no,Max_iteration,lb ,ub,dim,fitness)
Best_pos(1, 2) = round(Best_pos(1, 2));   
best_hd  = Best_pos(1, 2); % 批量处理
best_lr= Best_pos(1, 1);% 最佳初始学习率
best_l2 = Best_pos(1, 3);% 最佳L2正则化系数     
%%  构造网络结构
layers = [
 imageInputLayer([f_, 1])     % 输入层 输入数据规模[f_, 1, 1]
 
 convolution2dLayer([3, 1], 32)  % 卷积核大小 3*1 生成16张特征图
 batchNormalizationLayer         % 批归一化层
 reluLayer                       % Relu激活层
 
 convolution2dLayer([3, 1], 32)  % 卷积核大小 3*1 生成32张特征图
 batchNormalizationLayer         % 批归一化层
 reluLayer                       % Relu激活层

 dropoutLayer(0.1)               % Dropout层
 fullyConnectedLayer(num_class)          % 全连接层
 softmaxLayer
 classificationLayer];               % 分类层

%%  参数设置
options = trainingOptions('sgdm', ...     % SGDM 梯度下降算法
    'MiniBatchSize', best_hd,...               % 批大小,每次训练样本个数30
    'MaxEpochs', 100,...                  % 最大训练次数 500
    'InitialLearnRate', best_lr,...          % 初始学习率为0.01
    'LearnRateSchedule', 'piecewise',...  % 学习率下降
    'LearnRateDropFactor', 0.1,...        % 学习率下降因子 0.1
    'LearnRateDropPeriod', 80,...        % 经过400次训练后 学习率为 0.01*0.1
    'Shuffle', 'every-epoch',...          % 每次训练打乱数据集
    'L2Regularization', best_l2, ...         % L2正则化参数
    'Plots', 'training-progress',...      % 画出曲线
    'Verbose', false);

%%  训练模型
net = trainNetwork(p_train, t_train, layers, options);
inputSize = net.Layers(1).InputSize;

%% 数据处理
train =  p_train;
test  = p_test;

%% 提取高维特征
layer = 'fc';
P_train = activations(net,train,layer,'OutputAs','rows');
P_test  = activations(net,test,layer,'OutputAs','rows');

%% 设置输出
T_train = t_train;
T_test  = t_test;

%% 创建/训练SVM模型
T_train = string(T_train);
T_test = string(T_test);

T_train = double(T_train); P_train = double(P_train); 
T_test  = double(T_test) ; P_test = double(P_test); 

cmd = [' -t 2',' -c ',num2str(100),' -g ',num2str(0.01)];
model = svmtrain(T_train,P_train,cmd);

%% SVM仿真测试
T_sim1 = svmpredict(T_train,P_train,model);
T_sim2 = svmpredict(T_test,P_test,model);
%% 迭代图
figure
plot(curve, 'linewidth',1.5);
title('适应度')
xlabel('The number of iterations')
ylabel('Fitness')
%% 计算准确率
accuracy1 = mean(T_sim1 == T_train) * 100;
accuracy2 = mean(T_sim2 == T_test) * 100;

%% 打印
disp(['训练集准确率：' num2str(accuracy1) '%'] )
disp(['测试集准确率：' num2str(accuracy2) '%'] )



%% 格式转换
T_sim1 = double(T_sim1); 
T_sim2 = double(T_sim2); 

M = length(T_train);
N = length(T_test);

%%  绘图
figure
plot(1:M,T_train,'r-*',1:M,T_sim1,'b-o','LineWidth',1)
legend('真实值','SSA-CNN-SVM预测值')
xlabel('预测样本')
ylabel('预测结果')
string={'训练集预测结果对比';['准确率=' num2str(accuracy1) '%']};
title(string)
grid

figure
plot(1:N,T_test,'r-*',1:N,T_sim2,'b-o','LineWidth',1)
legend('真实值','SSA-CNN-SVM预测值')
xlabel('预测样本')
ylabel('预测结果')
string={'测试集预测结果对比';['准确率=' num2str(accuracy2) '%']};
title(string)
grid

%%  混淆矩阵
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
function fitness = fical(x)
%%  从主函数中获取训练数据
    f_ = evalin('base', 'f_');
    num_class = evalin('base', 'num_class');
    p_train = evalin('base', 'p_train');
    t_train = evalin('base', 't_train');
    T_train = evalin('base', 'T_train');
    
    
best_hd  = round(x(1, 2)); % 批量处理
best_lr= x(1, 1);% 最佳初始学习率
best_l2 = x(1, 3);% 最佳L2正则化系数
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
    'Plots', 'none',...      % 画出曲线
    'Verbose', false);

%%  训练模型
net = trainNetwork(p_train, t_train, layers, options);
inputSize = net.Layers(1).InputSize;

%% 数据处理
train =  p_train;

%% 提取高维特征
layer = 'fc';
P_train = activations(net,train,layer,'OutputAs','rows');

%% 设置输出
T_train = t_train;

%% 创建/训练SVM模型
T_train = string(T_train);

T_train = double(T_train); P_train = double(P_train); 

cmd = [' -t 2',' -c ',num2str(100),' -g ',num2str(0.01)];
model = svmtrain(T_train,P_train,cmd);

%% SVM仿真测试
T_sim1 = svmpredict(T_train,P_train,model);

%% 计算准确率
accuracy1 = mean(T_sim1 == T_train) ;

%% 性能评价
fitness = 1 - accuracy1 ;

end
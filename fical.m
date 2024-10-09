function fitness = fical(x)
%%  ���������л�ȡѵ������
    f_ = evalin('base', 'f_');
    num_class = evalin('base', 'num_class');
    p_train = evalin('base', 'p_train');
    t_train = evalin('base', 't_train');
    T_train = evalin('base', 'T_train');
    
    
best_hd  = round(x(1, 2)); % ��������
best_lr= x(1, 1);% ��ѳ�ʼѧϰ��
best_l2 = x(1, 3);% ���L2����ϵ��
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
    'Plots', 'none',...      % ��������
    'Verbose', false);

%%  ѵ��ģ��
net = trainNetwork(p_train, t_train, layers, options);
inputSize = net.Layers(1).InputSize;

%% ���ݴ���
train =  p_train;

%% ��ȡ��ά����
layer = 'fc';
P_train = activations(net,train,layer,'OutputAs','rows');

%% �������
T_train = t_train;

%% ����/ѵ��SVMģ��
T_train = string(T_train);

T_train = double(T_train); P_train = double(P_train); 

cmd = [' -t 2',' -c ',num2str(100),' -g ',num2str(0.01)];
model = svmtrain(T_train,P_train,cmd);

%% SVM�������
T_sim1 = svmpredict(T_train,P_train,model);

%% ����׼ȷ��
accuracy1 = mean(T_sim1 == T_train) ;

%% ��������
fitness = 1 - accuracy1 ;

end
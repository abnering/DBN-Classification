function test_example_DBN
%load mnist_uint8;

% train_x = double(train_x) / 255;
% test_x  = double(test_x)  / 255;
% train_y = double(train_y);
% test_y  = double(test_y);
% train_x = importdata('E:\深度学习动物识别\bigtrain\trainSet1.csv');
% train_y = importdata('E:\深度学习动物识别\bigtrain\trainLabel1.csv');
train_x = importdata('trainSet.csv');
train_y = importdata('trainLabel.csv');
test_x = importdata('testSet.csv');
test_y = importdata('testLabel.csv');

train_x = double(train_x(1:5000,:));
train_y = train_y(1:5000,:);

% test_x = [train_x(1:2500,:);train_x(22501:25000,:)];
% test_y = [train_y(1:2500,:);train_y(22501:25000,:)];
% 
% train_x = train_x(2501:22500,:);
% train_y = train_y(2501:22500,:);





%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
rand('state',0)
%train dbn
dbn.sizes = [200 100  ];
opts.numepochs =   40;
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   0.1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 2);
nn.activation_function = 'sigm';

%train nn
opts.numepochs =  300;
opts.batchsize = 100;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);
er
save bad;

assert(er < 0.10, 'Too big error');

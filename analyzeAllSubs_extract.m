   
clc; clear;
%% DATABASE II
    Subjects = ["S02_restingPre_EC.mat", 
               "S03_restingPre_EC.mat", 
               "S04_restingPre_EC.mat", 
               "S05_restingPre_EC.mat", 
               "S06_restingPre_EC.mat", 
               "S07_restingPre_EC.mat", 
               "S08_restingPre_EC.mat", 
               "S09_restingPre_EC.mat", 
               "S10_restingPre_EC.mat", 
               "S11_restingPre_EC.mat"];
   
    
   Subjects2 = ["S02_restingPre_EO.mat",
                "S03_restingPre_EO.mat",
                "S04_restingPre_EO.mat",
                "S05_restingPre_EO.mat",
                "S06_restingPre_EO.mat",
                "S07_restingPre_EO.mat",
                "S08_restingPre_EO.mat",
                "S09_restingPre_EO.mat",
                "S10_restingPre_EO.mat", 
                "S11_restingPre_EO.mat"];
            
%% DATABASE I
% A copy of thesame dataset is found at the link below qhich shall be used
% to read the data

% link: https://datahub.io/machine-learning/eeg-eye-state

data  = webread("https://datahub.io/machine-learning/eeg-eye-state/r/eeg-eye-state.csv");

%convert data table read from web to matrix so as to remove the string headers.
data = data{:,:};

%read data size, r stands for the row count and c stands for columns(Channel)
[r c] = size(data);

%% DATABASE I PRE-PROCESSING SO IT WILL BE USED FOR TESTING AS WELL on 
%   A network trained on 1-6 Subjects from DABASE I

%dat = rmoutliers(data,'mean');
dat = data;

% vec = [3.4 4.6 56.2; 23.5 3.1 4.3; 5.4 -12.3 6.7];
% vas = mean(delta(vec, 2));
% disp(vas);
% vas = mean(delta(vec, 2));
% disp(vas);
% vas = mean(beta(vec, 2));
% disp(vas);
% vas = mean(theta(vec, 2));
% disp(vas);
% now check the size of the new dataset rr:row count and cc:columns(Channel)
[rr cc]  = size(dat)

%now get the labels from the 15th column
label = dat(:,15);

label(label==0) = 2;

% now get the data and leave out the label
da = dat(:,[1:14]);

% normalize the data
da = delta(zscore(da), 128);

%Get the boudary for slicing the data into epochs
o = floor(length(label)/25);
% Now for a 4D dimension of data that is required by CNN
% load data in 4D
mydata = [];
for i=1:o
        mydata(:,:,:,i) = da([i:i+24],:);
        mylab(i) = mode(label([i:i+24]));
end



%GET ALL INDICES
ch = [1:length(mylab)];

%split data accordingly

%FOR 4D data arrangement
TestDataI = mydata(:,:,:,ch);
testLabelI = label(ch);


%% TRAINING DATASET
ko = 1;
TrainAll=[];
TrainlabelAll =[];

for m=1:6 %length(Subjects)
    
    disp("Analysis on Subject "+ m)
    DatII = load(Subjects(m));

    DAII = DatII.dataRest([1:64],:);



    %Select 14 channels as it correspond to Dataset I
    %The 14 channel are labeled as AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
    mydaII = DAII([3, 7, 5, 9, 15, 23, 27, 64, 60, 52, 44, 40, 42, 36],:)';
    %myda = rmoutliers(mydaII,'mean');
    myda = zscore(mydaII);

    myda = delta(myda, 256);
    %form 2 dimension data with 100 millisecond window apiece
    % 256/10 = 25 approximatly.
    o = floor(length(myda)/25);
    mydataII = [];
    for i=ko: m*1536
            mydataII(:,:,:,i) = myda([i:i+24],[1:14]);
            mylab(i) = 1;
    end
    %disp(mydataII(:,:,:,1));
      
    DatIIS2EO = load(Subjects2(m));

    DAIIS2EO = DatIIS2EO.dataRest([1:64],:);



    %Select 14 channels as it correspond to Dataset I
    %The 14 channel are labeled as AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
    mydaIIS2EO = DAIIS2EO([3, 7, 5, 9, 15, 23, 27, 64, 60, 52, 44, 40, 42, 36],:)';
    %myda = rmoutliers(mydaII,'mean');
    mydaS2EO = zscore(mydaIIS2EO);

    mydaS2EO = delta(mydaS2EO, 256);
    %form 2 dimension data with 100 millisecond window apiece
    % 256/10 = 25 approximatly.
    oS2EO = floor(length(mydaS2EO)/25);
    mydataIIS2EO = [];
    for i=ko:oS2EO
            mydataIIS2EO(:,:,:,i) = mydaS2EO([i:i+24],[1:14]);
            mylabS2EO(i) = 2;
    end
    %%
    %Form dataset from both eye close and eye open datasets

    count1=0;
    count2 = 0;
    alldata = [];
    allLabels = [];
    
    % SANDWITCH CLOSE AND OPEN SLICED WINDOW INBETWEEN EACH OTHER 
    for b = 1:2*oS2EO

        if mod(b, 2) == 0
            count2 = count2 + 1;
             alldata(:,:,:,b) = mydataIIS2EO(:,:,:,count2);
             allLabels(b) = 2;
        else
            count1 = count1 + 1;
            alldata(:,:,:,b) = mydataII(:,:,:,count1);
            allLabels(b) = 1;
        end

    end
    
    
    if m == 1
         TrainAll(:,:,:,:) = alldata;
         TrainlabelAll = allLabels;
    else
         j = length(TrainAll);
         jj = length(alldata);
         TrainAll(:,:,:,j+1: j + jj) = alldata;
         TrainlabelAll(:,j+1:j + jj) = allLabels;      
    end
    
end

% Testing dataset arrangement 
ko = 1;
TestAll=[];
TestlabelAll =[];

TestSub7 = [];
TestLab7 = [];
TestSub8 = [];
TestLab8 = [];
TestSub9 = [];
TestLab9 = [];
TestSub10 = [];
TestLab10 = [];

for m=7:length(Subjects)
    
    disp("Analysis on Subject "+ m)
    DatII = load(Subjects(m));

    DAII = DatII.dataRest([1:64],:);



    %Select 14 channels as it correspond to Dataset I
    %The 14 channel are labeled as AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
    mydaII = DAII([3, 7, 5, 9, 15, 23, 27, 64, 60, 52, 44, 40, 42, 36],:)';
    %myda = rmoutliers(mydaII,'mean');
    myda = zscore(mydaII);

    myda = delta(myda, 256);
    
    %form 2 dimension data with 100 millisecond window apiece
    % 256/10 = 25 approximatly.
    o = floor(length(myda)/25);
    mydataII = [];
    for i=ko: m*1536
            mydataII(:,:,:,i) = myda([i:i+24],[1:14]);
            mylab(i) = 1;
    end

    %disp(mydataII(:,:,:,1));
    %%     
    DatIIS2EO = load(Subjects2(m));

    DAIIS2EO = DatIIS2EO.dataRest([1:64],:);



    %Select 14 channels as it correspond to Dataset I
    %The 14 channel are labeled as AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
    mydaIIS2EO = DAIIS2EO([3, 7, 5, 9, 15, 23, 27, 64, 60, 52, 44, 40, 42, 36],:)';
    %myda = rmoutliers(mydaII,'mean');
    mydaS2EO = zscore(mydaIIS2EO);

    mydaS2EO = delta(mydaS2EO, 256);
    %form 2 dimension data with 100 millisecond window apiece
    % 256/10 = 25 approximatly.
    oS2EO = floor(length(mydaS2EO)/25);
    mydataIIS2EO = [];
    for i=ko:oS2EO
            mydataIIS2EO(:,:,:,i) = mydaS2EO([i:i+24],[1:14]);
            mylabS2EO(i) = 2;
    end
    
    %Form dataset from both eye close and eye open datasets
    % SANDWITCH CLOSE AND OPEN SLICED WINDOW INBETWEEN EACH OTHER 
    count1=0;
    count2 = 0;
    alldata = [];
    allLabels = [];
    for b = 1:2*oS2EO

        if mod(b, 2) == 0
            count2 = count2 + 1;
             alldata(:,:,:,b) = mydataIIS2EO(:,:,:,count2);
             allLabels(b) = 2;
        else
            count1 = count1 + 1;
            alldata(:,:,:,b) = mydataII(:,:,:,count1);
            allLabels(b) = 1;
        end

    end
    
    if m == 7
         TestAll(:,:,:,:) = alldata;
         TestlabelAll = allLabels;
    else
         j = length(TestAll);
         jj = length(alldata);
         TestAll(:,:,:,j+1: j + jj) = alldata;
         TestlabelAll(:,j+1:j + jj) = allLabels;      
    end
    
    
    if m == 7
         TestSub7(:,:,:,:) = alldata;
         TestLab7 = allLabels;
    elseif m==8
         TestSub8(:,:,:,:) = alldata;
         TestLab8 = allLabels;
    elseif m == 9
         TestSub9(:,:,:,:) = alldata;
         TestLab9 = allLabels; 
    elseif m == 10
         TestSub10(:,:,:,:) = alldata;
         TestLab10 = allLabels;    
    end  
    
end

    %%
    %Four Layer CNN model
    %filter initial size
    fsize = 1;
    counter = 0;
    %filter initial quantity = number of channels 
    fcount = 1; %size(myda, 2);
    %set while loop stopper
    stopme = false;

    disp("Four layer evaluation");

 % for kl=6:10
     
 %   disp("Testing on subject "+ kl);
    
%     if kl==6
%         TestAll = TestDataI;
%         TestlabelAll = testLabelI;
%     elseif kl==7
%         TestAll = TestSub7;
%         TestlabelAll = TestLab7;
%     elseif kl==8
%       TestAll = TestSub8;
%         TestlabelAll = TestLab8;
%     elseif kl==9
%          TestAll = TestSub9;
%         TestlabelAll = TestLab9;
%     elseif kl==10
%          TestAll = TestSub10;
%          TestlabelAll = TestLab10;
%     end
    
    for fsize = 1:10
    
        for fcount=1:10

        layers = [
                imageInputLayer([25 14 1]) % 14X1X1 refers to number of features per sample = 14 channels of EEG data.
                convolution2dLayer(fsize, fcount,'Padding', 'same')
                reluLayer                 % add non linearity to the network
                maxPooling2dLayer(1,'stride',1)
                convolution2dLayer(fsize, fcount,'Padding','same') % Second layer starts
                reluLayer
                maxPooling2dLayer(1,'stride',1)
                convolution2dLayer(fsize, fcount,'Padding','same') % Third layer starts
                reluLayer
                maxPooling2dLayer(1,'stride',1)
                convolution2dLayer(fsize, fcount,'Padding','same') % Fourth layer starts
                reluLayer
                maxPooling2dLayer(1,'stride',1)
                %dropoutLayer(0.5)
                fullyConnectedLayer(60)  % A fully connected layer multiplies the input by a weight matrix and then adds a bias vector.
                fullyConnectedLayer(30)
                fullyConnectedLayer(2) % 2 refers to number of neurons in next output layer (number of output classes)
                softmaxLayer           % used for classification decision
                classificationLayer];


               %now form the arguments to be passed to the nework  
                args.label = TrainlabelAll; args.imgs = TrainAll;
                args.layers = layers; 
                options = trainingOptions('sgdm','MaxEpochs',50,'Verbose',false);
                args.kfold=10; args.options = options;
                %CNN4 = cnnKfold(args);  % cnnKfold function inside which the CNN is implemented and kfold used for validataion
                net = trainNetwork(TrainAll, categorical(TrainlabelAll), layers, options);


                Networks(fsize).net = net;
                Pd=classify(net, TestAll);
                con=confusionmat(categorical(TestlabelAll), Pd);
                acc = 100*sum(diag(con))/sum(con(:));
                disp(con);
                %comps = [fsize, fcount, acc], 1, 1;
                disp("ALL Subs Dsplay: " + fsize +", " + fcount + ", " + acc);
                %darray(:,:,:,fsize) =  comps;  % use to save used size, fcount and accuracies produced. 
                %Check net with Sub 7
                Pd=classify(net, TestSub7);
                con=confusionmat(categorical(TestLab7), Pd);
                acc = 100*sum(diag(con))/sum(con(:));
                disp(con);
                %comps = [fsize, fcount, acc], 1, 1;
                disp("Sub 7 Dsplay: " + fsize +", " + fcount + ", " + acc);
                
                %Check net with Sub 8
                Pd=classify(net, TestSub8);
                con=confusionmat(categorical(TestLab8), Pd);
                acc = 100*sum(diag(con))/sum(con(:));
                disp(con);
                %comps = [fsize, fcount, acc], 1, 1;
                disp("Sub 8 Dsplay: " + fsize +", " + fcount + ", " + acc);
                
                
                %Check net with Sub 9
                Pd=classify(net, TestSub9);
                con=confusionmat(categorical(TestLab9), Pd);
                acc = 100*sum(diag(con))/sum(con(:));
                disp(con);
                %comps = [fsize, fcount, acc], 1, 1;
                disp("Sub 9 Dsplay: " + fsize +", " + fcount + ", " + acc);
                
                %Check net with Sub 10
                Pd=classify(net, TestSub10);
                con=confusionmat(categorical(TestLab10), Pd);
                acc = 100*sum(diag(con))/sum(con(:));
                disp(con);
                %comps = [fsize, fcount, acc], 1, 1;
                disp("Sub 10 Dsplay: " + fsize +", " + fcount + ", " + acc);
                
          
        end
     end
 % end
 
 return;
    %%
%     function CNN = cnnKfold(args)
% 
%     fold=cvpartition(args.label,'kfold',args.kfold);
%     Afold=zeros(args.kfold,1); confmat=0;
%     for i=1:args.kfold
%       trainIdx=fold.training(i); testIdx=fold.test(i);
%       xtrain=args.imgs(:,:,1,trainIdx); ytrain=args.label(trainIdx);
%       xtest=args.imgs(:,:,1,testIdx); ytest=args.label(testIdx);
%       ytrain=categorical(ytrain); ytest=categorical(ytest);
%       net=trainNetwork(xtrain,ytrain,args.layers,args.options);
% 
%       Pred=classify(net,xtest);
%       con=confusionmat(ytest,Pred);
%       confmat=confmat+con; 
%       Afold(i,1)=100*sum(diag(con))/sum(con(:));
% 
%     end
%     Acc=mean(Afold); 
%     CNN.fold=Afold; CNN.acc=Acc; CNN.con=confmat
%     CNN.net = net;
% 
%     fprintf('\n Val Classification Accuracy (CNN): %g %% \n ',Acc);
%     end

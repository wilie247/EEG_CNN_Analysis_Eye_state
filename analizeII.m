    %%
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
    %%
for m=9:length(Subjects)
    
    disp("Analysis on Subject "+ m)
    DatII = load(Subjects(m));

    DAII = DatII.dataRest([1:64],:);



    %Select 14 channels as it correspond to Dataset I
    %The 14 channel are labeled as AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
    mydaII = DAII([3, 7, 5, 9, 15, 23, 27, 64, 60, 52, 44, 40, 42, 36],:)';
    %myda = rmoutliers(mydaII,'mean');
    myda = zscore(mydaII);

    %form 2 dimension data with 100 millisecond window apiece
    % 256/10 = 25 approximatly.
    o = floor(length(myda)/25);
    mydataII = [];
    for i=1:o
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

    %form 2 dimension data with 100 millisecond window apiece
    % 256/10 = 25 approximatly.
    oS2EO = floor(length(mydaS2EO)/25);
    mydataIIS2EO = [];
    for i=1:oS2EO
            mydataIIS2EO(:,:,:,i) = mydaS2EO([i:i+24],[1:14]);
            mylabS2EO(i) = 2;
    end


    %%
    %Form dataset from both eye close and eye open datasets

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

    %%
    %for a train - test split with temporal ordering in place.
    %create indices
    v = [1:size(allLabels, 2)];
    shuffle = v;
    c = .6 * size(allLabels, 2);
    c = round(c);

    count = shuffle([1:c]);
    count2 = shuffle([c+1:size(shuffle, 2)]);
    %split data accordingly

    %FOR 4D data arrangement
    TrainData = alldata(:,:,:,count);
    trainLabel = allLabels(count);
    TestData = alldata(:,:,:,count2);
    testLabel = allLabels(count2);
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
                args.label = trainLabel; args.imgs = TrainData;
                args.layers = layers; 
                options = trainingOptions('sgdm','MaxEpochs',50,'Verbose',false);
                args.kfold=10; args.options = options;
                %CNN4 = cnnKfold(args);  % cnnKfold function inside which the CNN is implemented and kfold used for validataion
                net = trainNetwork(TrainData, categorical(trainLabel), layers, options);


                counter = counter + 1;

                Networks(counter).net = net;

                Pd=classify(net, TestData);

                con=confusionmat(categorical(testLabel), Pd);

                acc = 100*sum(diag(con))/sum(con(:));

                %acc = CNN.acc;

                 disp(con);
                comps = [fsize, fcount, acc], 1, 1;
                 disp("Dsplay: " + fsize +", " + fcount + ", " + acc);
                 darray(:,:,:,counter) =  comps;  % use to save used size, fcount and accuracies produced.
          
                  % stopme = true;
                  % break;
             end

        end
 end
  
    %%
    function CNN = cnnKfold(args)

    fold=cvpartition(args.label,'kfold',args.kfold);
    Afold=zeros(args.kfold,1); confmat=0;
    for i=1:args.kfold
      trainIdx=fold.training(i); testIdx=fold.test(i);
      xtrain=args.imgs(:,:,1,trainIdx); ytrain=args.label(trainIdx);
      xtest=args.imgs(:,:,1,testIdx); ytest=args.label(testIdx);
      ytrain=categorical(ytrain); ytest=categorical(ytest);
      net=trainNetwork(xtrain,ytrain,args.layers,args.options);

      Pred=classify(net,xtest);
      con=confusionmat(ytest,Pred);
      confmat=confmat+con; 
      Afold(i,1)=100*sum(diag(con))/sum(con(:));

    end
    Acc=mean(Afold); 
    CNN.fold=Afold; CNN.acc=Acc; CNN.con=confmat
    CNN.net = net;

    fprintf('\n Val Classification Accuracy (CNN): %g %% \n ',Acc);
    end

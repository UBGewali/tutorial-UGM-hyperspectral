run('./startup.m')

%dataset_name = {'IP', 'PaviaU', 'PaviaC', or 'Salinas'}
dataset_name = 'IP';
[image, groundTruth, oldLabelNames] = load_dataset(dataset_name);

[im_rows, im_cols, im_bands] = size( image );
spectra = reshape( image, [im_rows*im_cols,im_bands] );
spectra = zscore(spectra,[],1);
image = reshape( spectra, [im_rows, im_cols, im_bands] );


numOfMinDataPts = 200;
numOfTrials = 30; 
numTestEg = 50;
numOfLabels = max( groundTruth(:) );


labelNames = {};
labelCnt = 1;
for i = 1:numOfLabels
    locations = find(groundTruth==i);
    numOfDataPts = length( locations );
    if numOfDataPts < numOfMinDataPts
        groundTruth( locations ) = 0;
    else
        groundTruth( locations ) = labelCnt;
        labelNames{labelCnt} = oldLabelNames{i};
        labelCnt = labelCnt + 1;
    end
end
numOfLabels = labelCnt - 1;


%generate featuers
classifiers = {@() LR(),@() RF(),@() SAM(),@() SVM_SE(),@() SVM_ESAM,@() GP_SE(), @() GP_ESAM};
classifiers_name = {'LR','RF','SAM','SVM_SE','SVM_ESAM','GP_SE','GP_ESAM'};
disp( 'Computing features!!!' );
features = {RAW(image),EMP(image)};
features_name = {'RAW','EMP'};
postprocess_name = {'None','MRF','CRF'};

disp( 'Features computed!!!' );

numTrainEgL = [20,40,60,80,100];
for t = 1:length(numTrainEgL)
    numTrainEg = numTrainEgL(t);
    result = cell(1,numOfTrials);
    for i = 1:numOfTrials
        XtrainC = [];
     	Ytrain = [];
    	XtestC = [];
    	Ytest = [];
    	XvalidC = [];
    	Yvalid = [];
 
        for j = 1:numOfLabels
             [r,c] = find( groundTruth==j);
    	     numOfDataPts = length(r);
	         idx = randperm(numOfDataPts,numTestEg+numTrainEg);
	         r = r(idx); c = c(idx);	
             [train_val, test] = crossvalind('LeaveMOut', length(r), numTestEg);
	         test = find(test);
	         train_val = find(train_val);
	         XtestC = [XtestC; [r(test),c(test)] ];
	         Ytest = [Ytest; j*ones(length(test),1)];	
	         train = train_val(1:round(.7*numTrainEg));
	         valid = train_val((round(.7*numTrainEg)+1):end);
	         XtrainC = [XtrainC; [r(train),c(train)] ];
	         Ytrain = [Ytrain; j*ones(length(train),1)];	
	         XvalidC = [XvalidC; [r(valid),c(valid)] ];
	         Yvalid = [Yvalid; j*ones(length(valid),1)];
        end
	
	    %features
	    for j = 1:length(features)
		    %classifiers
		    for k = 1:length(classifiers)
			    [out_maps,time_taken] = classify(classifiers{k},features{j},...
				   XtrainC,Ytrain,XvalidC,Yvalid);
			
			    for l = 1:length(postprocess_name)			
				    out_map = out_maps{l};
				    Ypred = out_map( sub2ind([im_rows,im_cols],XtestC(:,1),XtestC(:,2)) );
				    OA = nnz(Ypred==Ytest)/numel(Ypred);
				    confMat = confusionmat(Ytest,Ypred);
				    result{i}.(features_name{j}).(classifiers_name{k}).(postprocess_name{l}).time=time_taken(l);
				    result{i}.(features_name{j}).(classifiers_name{k}).(postprocess_name{l}).OA=OA;
				    result{i}.(features_name{j}).(classifiers_name{k}).(postprocess_name{l}).confMat=confMat;
                end 				
			    disp( [ features_name{j},'-', classifiers_name{k},' done!'] );
            end		
	    end	

    end	
    %saving results
    save( ['results/result_',dataset_name,'_',num2str(numTrainEg),'.mat'], 'result' );
end

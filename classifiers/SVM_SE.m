classdef SVM_SE < handle
    properties
        model;
	numOfClasses;
    end
    methods 
        function obj = train(obj, Xtrain, Ytrain)
		obj.numOfClasses = max(Ytrain);
		params = [0.001,0.01,0.1,1,10,100,1000];
		
		numOfParams = length(params);	
		numOfSVMPara = 2;
		idxList = cell(1,numOfSVMPara);
    		x = repmat({1:numOfParams},[1, numOfSVMPara]);
    		[idxList{:}] = ndgrid(x{:});
    		combs = [];
    		for i = 1:length(idxList)
			combs = [combs,idxList{i}(:)];
    		end

		numParams = size(combs,1);
		
		Xt = [];
		Yt = [];
		Xv = [];
		Yv = [];
		
		for i = 1:obj.numOfClasses
			idx = Ytrain==i;
			idx = find(idx);
			m =  length(idx);
			m_mid = round(0.8*m);
			idx_t = idx(1:m_mid);
			idx_v = idx((m_mid+1):end);
			Xt = [Xt; Xtrain(idx_t,:)];	
			Yt = [Yt; Ytrain(idx_t,:)];	
			Xv = [Xv; Xtrain(idx_v,:)];	
			Yv = [Yv; Ytrain(idx_v,:)];	
		end
				
		perfL = zeros(numParams,1);

		for i = 1:numParams
			scale = params(combs(i,1));
			C = params(combs(i,2));
			model = svmtrain(Yt, Xt, ['-b 1 -t 2 -q -g ', num2str(scale), ' -c ', num2str(C)]);
			[~,acc,~] =  svmpredict(Yv,Xv,model, '-b 1 -q' );
			perfL(i) = acc(1);	
		end
		[~,idx] = max(perfL);
		scale = params(combs(idx,1));
		C = params(combs(idx,2));
		obj.model = svmtrain(Ytrain, Xtrain, ['-b 1 -t 2 -q -g ', num2str(scale), ' -c ', num2str(C)]);
	end
	function [preds,prob] = predict(obj, Xtest)
		[~,~,prob] = svmpredict(ones(size(Xtest,1),1), Xtest, obj.model, '-b 1 -q');
		order = obj.model.Label;
		prob = prob';
		prob = prob(order,:);
    		[~,preds] = max(prob);
	end

	function [Yout,Yprobs] = predictImg(obj, img)
		[m,n,p] = size(img);
    		spectra = reshape( img, [m*n,p] );
    		[preds,probs] = obj.predict( spectra);
 		Yout = reshape( preds, [m,n] );
		Yprobs = zeros(m,n,obj.numOfClasses);
    		for i = 1:obj.numOfClasses
        		prob_img = reshape( probs(i,:), [m,n] );
			Yprobs(:,:,i) = prob_img;
    		end 
	end	
    end
end

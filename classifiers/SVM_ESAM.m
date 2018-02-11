classdef SVM_ESAM < handle
    properties
        model;
	numOfClasses;
	Xtrain;
	scale;
	C;
    end
    methods 
	function K = ESAM(obj,x,z,scale)
    		xz = x*z';
    		xnorm = sqrt(sum( x.^2,2) );
    		znorm = sqrt(sum( z.^2,2) );
    		K = xz ./ (xnorm*znorm'+1e-10);
    		K = max( K, -1 );
    		K = min( K, 1 );
    		K = exp(-acos(K)/scale);    
    		K = real(K);
	end
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
		models = cell(numParams,1);

		for i = 1:numParams
			scale = params(combs(i,1));
			C = params(combs(i,2));
			K = obj.ESAM(Xt,Xt,scale);
			K = [(1:size(Xt,1))', K];
			model = svmtrain(Yt, K, ['-t 4 -b 1 -q -c ',num2str(C)]);
			K = obj.ESAM(Xv,Xt,scale);
			K = [(1:size(Xv,1))', K];
			[~,acc,~] =  svmpredict(Yv,K,model, '-b 1 -q' );
			perfL(i) = acc(1);	
		end
		[~,idx] = max(perfL);
		scale = params(combs(i,1));
		C = params(combs(i,2));
		K = obj.ESAM(Xtrain,Xtrain,scale);
		K = [(1:size(Xtrain,1))', K];
		obj.model = svmtrain(Ytrain, K, ['-t 4 -b 1 -q -c ',num2str(C)]);
		obj.Xtrain = Xtrain;
		obj.scale = scale;
		obj.C = C;
	end
	function [preds,prob] = predict(obj, Xtest)
		K = obj.ESAM(Xtest,obj.Xtrain,obj.scale);
		K = [(1:size(Xtest,1))', K];
		[~,~,prob] = svmpredict(ones(size(Xtest,1),1), K, obj.model, '-b 1 -q');
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

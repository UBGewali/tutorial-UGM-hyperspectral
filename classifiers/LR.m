classdef LR < handle
    properties
        model;
	numOfClasses;
    end
    methods 
        function obj = train(obj, Xtrain, Ytrain)
		params = [0.001,0.01,0.1,1,10,100,1000];
		numParams = length(params);
		Xt = [];
		Yt = [];
		Xv = [];
		Yv = [];
		obj.numOfClasses = max(Ytrain);	
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
			C = params(i);
			models = train(Yt, sparse(Xt), ['-s 0 -B 1 -q -c ', num2str(C)]);
			[l,acc,p] =  predict(Yv,sparse(Xv),models, '-b 1 -q' );
			perfL(i) = acc(1);	
		end
		[~,idx] = max(perfL);
		C = params(idx);
		obj.model = train(Ytrain, sparse(Xtrain), ['-s 0 -B 1 -q -c ', num2str(C)]);
	end


	function [Yout,Yprob] = predict(obj, Xtest)
		[~,~,prob] = predict(ones(size(Xtest,1),1), sparse(Xtest), obj.model, '-b 1 -q');
		order = obj.model.Label;
		prob = prob';
		Yprob = prob(order,:);
		[~,Yout] = max(Yprob);
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

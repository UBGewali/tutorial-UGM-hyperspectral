classdef RF < handle
	properties
		model;
	end
	methods
		function obj = train(obj, Xtrain, Ytrain)
			numOfTrees = [50,100,200,400];
			perfList = zeros(size(numOfTrees));
			Xt = [];
			Yt = [];
			Xv = [];
			Yv = [];
			numOfClasses = max(Ytrain);
			for i = 1:numOfClasses
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
			for i = 1:length(numOfTrees)
				B = TreeBagger(numOfTrees(i),Xt,Yt);
				[preds, ~] = predict(B, Xv);
    				preds = cellfun(@str2num, preds);
			 	perfList = sum(preds==Yv);
			end
			[~,idx] = max(perfList);
			obj.model = TreeBagger(numOfTrees(idx),Xtrain,Ytrain);
		end	

		function [Yout,Yprob] = predict(obj, Xtest)
    			[preds, probs] = predict(obj.model, Xtest);
    			Yout = cellfun(@str2num, preds)';	
    			className = cellfun(@str2num,obj.model.ClassNames);
			Yprob = zeros(size(probs));
			Yprob(:,className) = probs;
		end

		function [Yout,Yprobs] = predictImg(obj, img)
			[m,n,p] = size(img);
    			spectra = reshape( img, [m*n,p] );
    			[preds, probs] = predict(obj.model, spectra);
    			preds = cellfun(@str2num, preds);
    			className = cellfun(@str2num,obj.model.ClassNames);

    			Yout = reshape( preds, [m,n] ); 
   			numOfLabels = size(probs,2);
			Yprobs = zeros(m,n,numOfLabels);
    			for i = 1:numOfLabels
        			prob_img = reshape( probs(:,i), [m,n] );
				Yprobs(:,:,className(i)) = prob_img;
    			end
		end
	end
end

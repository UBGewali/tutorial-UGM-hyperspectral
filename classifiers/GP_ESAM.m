classdef GP_ESAM < handle
    properties
        model;
	numOfClasses;
	params;
	vcomb;
	comb;

	meanfunc = @meanZero;
	covfunc = @covESAM;
	likfunc = @likErf;
	infmethod = @infLaplace;

    end
    methods 
        function obj = train(obj, Xtrain, Ytrain)
		obj.numOfClasses = max(Ytrain);
		obj.comb = nchoosek(obj.numOfClasses,2);
		obj.params = cell(1,obj.comb);
		obj.vcomb = nchoosek(1:obj.numOfClasses,2);

		for k = 1:obj.comb
			i = obj.vcomb(k,1);
			j = obj.vcomb(k,2);

			idx = or(Ytrain==i,Ytrain==j);
			x = Xtrain(idx,:);
			y = Ytrain(idx,:);
			y = double(y == i);
			y( y==0 ) = -1;
							
			ell = 1;
			sf = 1;
			hyp.cov = log([ell sf]);

			hyp = minimize(hyp, @gp, -100, obj.infmethod, obj.meanfunc,obj.covfunc,...
		        obj.likfunc, x , y);
			obj.params{k}.x = x;
			obj.params{k}.y = y;
			obj.params{k}.hyp = hyp;	
		end
	end
	
	function [preds,prob] = predict(obj, Xtest)
		npts = size(Xtest,1);	
	
		tempD = zeros(obj.comb,npts);
		for k = 1:obj.comb
			i = obj.vcomb(k,1);
			j = obj.vcomb(k,2);
			for trys = 1:5
				try
					[a,b,c,d,lp] = gp(obj.params{k}.hyp,obj.infmethod, obj.meanfunc, obj.covfunc, obj.likfunc,...
						       obj.params{k}.x, obj.params{k}.y, Xtest, ones(npts,1) );
					break;
				catch
					obj.params{k}.hyp.cov(2) = log(exp(obj.params{k}.hyp.cov(2)) * 1.05); 	 
				end 
			end				
			tempD(k,:) = exp(lp);
		end
		D = zeros(obj.numOfClasses,obj.numOfClasses,npts);
		for k=1:obj.comb
    			i = obj.vcomb(k,1);
    			j = obj.vcomb(k,2);
    			D(i,j,:) = tempD(k,:);
    			D(j,i,:) = 1-D(i,j,:);
		end	
		prob = squeeze(cell2mat(...
           	cellfun(@get_onevsone, num2cell(D,[1 2]),'UniformOutput',false)));
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

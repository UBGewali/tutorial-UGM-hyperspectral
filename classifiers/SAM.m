classdef SAM < handle
    properties
        model;
	numOfClasses;
    end
    methods 
        function obj = train(obj, Xtrain, Ytrain)
		obj.numOfClasses = max(Ytrain);
		obj.model = cell(1,obj.numOfClasses);
		for i = 1:obj.numOfClasses
			obj.model{i} = Xtrain( (Ytrain==i) , : );
		end
	end

	function [Yout,angles] = predict(obj, Xtest)
		m = size(Xtest,1);
		angles = zeros(obj.numOfClasses,m);
		for i = 1:obj.numOfClasses
			dot_prod = obj.model{i}*Xtest';	
			a = sqrt(sum( obj.model{i}.^2, 2 ) );
			b = sqrt(sum( Xtest.^2, 2 ) );
			K = dot_prod ./ (a*b'+1e-10);
			K = max(K, -1);
			K = min(K, 1);
			K = acos(K);
			K = min(K,[],1);
			angles(i,:) = K;		
		end
		angles = exp(-angles);
		[~,Yout] = max(angles);
	end

	function [Yout,Yangs] = predictImg(obj, img)
		[m,n,p] = size(img);
    		spectra = reshape( img, [m*n,p] );
    		[preds,angs] = obj.predict( spectra);
 		Yout = reshape( preds, [m,n] );
		Yangs = zeros(m,n,obj.numOfClasses);
    		for i = 1:obj.numOfClasses
        		angs_img = reshape( angs(i,:), [m,n] );
			Yangs(:,:,i) = angs_img;
    		end 
	end	
    end
end

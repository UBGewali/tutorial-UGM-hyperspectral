classdef EMP < handle
	properties
		maps;	
	end
	methods
		function obj = EMP(image)
			percVarL = [ 0.84:0.05:0.99 ];
			numOfOptL = [ 2,4,8];
			structStepSizeL = [2,4,8];
			numOfMaps = length(percVarL)*length(numOfOptL)*length(structStepSizeL);
			obj.maps = cell(1,numOfMaps);
			count = 1;
    		        [m,n,p] = size(image);
    		        spectra = reshape( image, [m*n,p] );
    		        [~,comps,var] = pca(spectra);
			for i = 1:length(percVarL)
    			        idx = find( cumsum(var)/sum(var) > percVarL(i) );
    			        new_p = idx(1);
    			        PCA_spectra = comps(:,1:new_p);
    			        PCA_img = reshape( PCA_spectra, [m,n,new_p] );
				for j = 1:length(numOfOptL)
					for k = 1:length(structStepSizeL)
						obj.maps{count} = extendedMP(PCA_img,numOfOptL(j),structStepSizeL(k));
						count = count + 1;
					end
				end
			end

		end
		function n = length(obj)
			n = length(obj.maps);
		end
		function amap = get(obj,n)
			amap = obj.maps{n};
		end
	end
end



function EMP = extendedMP( PCA_img, numOfOp, structStepSize )
    structInitSize = 2;

    [m,n,new_p] = size(PCA_img); 
    EMP = zeros(m,n,new_p*2*(numOfOp+1));
    k=1;
    for i = 1:new_p
        structSize = structInitSize;
        EMP(:,:,k) = PCA_img(:,:,i);
        k = k + 1;
        for j = 1:numOfOp
            se = strel('disk', structSize, 0);
            EMP(:,:,k) = imopen(PCA_img(:,:,i),se);
            k = k + 1;
            EMP(:,:,k) = imclose(PCA_img(:,:,i),se);
            k = k + 1;
            structSize = structSize + structStepSize;
        end
    end
    
end

classdef PCA < handle
	properties
		maps;	
	end
	methods
		function obj = PCA(image)
			percVarL = [ 0.84:0.05:0.99 ];
			numOfMaps = length(percVarL);
			obj.maps = cell(1,numOfMaps);

    			[m,n,p] = size(image);
    			spectra = reshape( image, [m*n,p] );
    			[~,comps,var] = pca(spectra);
			for i = 1:length(percVarL)
    				idx = find( cumsum(var)/sum(var) > percVarL(i) );
    				new_p = idx(1);
    				PCA_spectra = comps(:,1:new_p);
    				obj.maps{i} = reshape( PCA_spectra, [m,n,new_p] );
				
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




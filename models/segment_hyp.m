function [segment,adj] = segment_hyp(img, numSupPixels)
    spatialPara = 100;
   
    [nrows,ncols,nbands] = size(img);
    spectra = reshape(img, [nrows*ncols, nbands]);
    spectra = zscore(spectra, []);
    img = single(reshape(spectra, [nrows,ncols,nbands])); 
    
    segment = double(vl_slic(img, round(sqrt((nrows*ncols)/numSupPixels)), spatialPara,'MinRegionSize',9));
    segment_img = zeros(size(segment));
	segment_idx = unique(segment);
	for k = 1:length(segment_idx)
	     segment_img(segment==segment_idx(k)) = k;
	end
	segment = segment_img;
	adj = make_graph(segment);
end



function adj = make_graph(segment_img)
    [nrows,ncols] = size(segment_img);
    num_nodes = max(segment_img(:));
    adj = zeros(num_nodes, num_nodes);
    for i = 1:num_nodes
        [r,c] = find(segment_img == i);
        for j = 1:length(r)
            dx = {[0,1],[1,0],[0,-1],[-1,0]};
            for k = 1:length(dx)
                r1 = r(j) + dx{k}(1);
                c1 = c(j) + dx{k}(2);
                if r1>0 && c1>0 && r1<=nrows && c1<=ncols
                    i_end = segment_img(r1,c1);
                    if i ~= i_end
                        adj(i,i_end) = 1;
                    end
                end
            end
        end
    end
end

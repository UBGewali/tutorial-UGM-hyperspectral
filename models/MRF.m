function out_map = MRF(probs,Xvalid,Yvalid)
    param1 = [0.01, 0.1, 1,10];
    num1 = length(param1);
    out_map_array = cell(1,num1); 
    perf_array = zeros(1,num1);	
    
    [nRows, nCols, nClasses] = size(probs);
    adj = make_grid_graph(nRows,nCols);
    for i = 1:num1	
    		out_map = MRFpred(probs, adj, param1(i));
    		Yout = out_map( sub2ind(size(out_map),Xvalid(:,1),Xvalid(:,2)) );
        	perf_array(i) = sum(Yout==Yvalid) / length(Yout); 
        	out_map_array{i} = out_map;
    end
    [~,idx] = max(perf_array);
    out_map = out_map_array{idx};

end



function pred = MRFpred(probs, adj, beta, img, alpha)
if nargin > 3
     useFeatures = true;
else
     useFeatures = false;
end


[nRows, nCols, nClasses] = size(probs);
nodePot = reshape(probs, [nRows*nCols, nClasses]);

if useFeatures
    [nRows, nCols, nBands] = size(img);
    spectra = reshape(img, [nRows*nCols, nBands]);
end


edgeStruct = UGM_makeEdgeStruct(adj,nClasses,0); %third argument=1 => use mex

edgePot = zeros(nClasses,nClasses,edgeStruct.nEdges);
for e = 1:edgeStruct.nEdges
   n1 = edgeStruct.edgeEnds(e,1);
   n2 = edgeStruct.edgeEnds(e,2);    
   if ~useFeatures 
       diffPot = beta;
   else
       diffPot = beta* exp(-(1-pdist( spectra([n1,n2],:) ,'cosine'))/alpha);
   end
   edgePot(:,:,e) = exp(-diffPot*(1-eye(nClasses)));
end

pred = double(UGM_Decode_AlphaExpansion(nodePot,edgePot,edgeStruct));
pred = reshape(pred, [nRows,nCols]);

end




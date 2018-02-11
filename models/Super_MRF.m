function out_map = Super_MRF(probs,img,numSupPix,Xvalid,Yvalid)
    [nrows,ncols,nclass] = size(probs);
    probs = reshape(probs,[nrows*ncols,nclass]);
    
    out_map = [];
    best_perf = -inf;    
    
    [segments,adj] = segment_hyp(img, numSupPix);
    num_segments = max(segments(:));
    node_prob = zeros(num_segments,nclass);
    for j = 1:num_segments
        node_idx = segments(:)==j;
	node_prob(j,:) = mean(probs(node_idx,:),1);
    end
    param1 = [0.01, 0.1, 1, 10];
    for j = 1:length(param1)
        y = apply_MRF(node_prob,adj, param1(j));
	y_map = zeros(nrows,ncols);
	for p = 1:num_segments
	    y_map(segments(:)==p) = y(p);
	end
    	Yout = y_map( sub2ind(size(y_map),Xvalid(:,1),Xvalid(:,2)) );
        perf = sum(Yout==Yvalid) / length(Yout); 
	if perf > best_perf
	        out_map = y_map;
	        best_perf = perf;
        end
     end
	
end

function yout = apply_MRF(unary_pot, adj, w)
    [num_nodes, num_classes] = size(unary_pot);
    edge_struct = UGM_makeEdgeStruct(adj, num_classes, 0);%third argument=1 => use mex
    edge_pot = zeros(num_classes, num_classes, edge_struct.nEdges);
    for e = 1:edge_struct.nEdges
	    edge_pot(:,:,e) = exp(-w*(1-eye(num_classes)));
    end
    yout = UGM_Decode_AlphaExpansion(unary_pot, edge_pot, edge_struct, @UGM_Decode_GraphCut);
    %yout = UGM_Decode_LinProg(unary_pot, edge_pot, edge_struct);
end


function [out_maps,time_taken] = classify( model, feat, Xt, Yt, Xv, Yv)
    numOfFeat = feat.length();
	perfL = zeros(1,numOfFeat);
    mdlL = cell(1,numOfFeat);

	tic();
	for i = 1:numOfFeat
		fimg = feat.get(i);
		[im_rows, im_cols, im_bands] = size( fimg );
		spectra = reshape( fimg, [im_rows*im_cols,im_bands] );
     		Xtrain = spectra( sub2ind([im_rows,im_cols],Xt(:,1),Xt(:,2)) ,:);
     		Xvalid = spectra( sub2ind([im_rows,im_cols],Xv(:,1),Xv(:,2)) ,:);
    		mdl = model();		
		mdl.train( Xtrain, Yt );
		[pred,~] = mdl.predict(Xvalid);
		perfL(i) = sum(pred' == Yv);
		mdlL{i} = mdl;
		
	end 
	[~,idx] = max(perfL);
	mdl = mdlL{idx};
	[out_image1,prob_map] = mdl.predictImg( feat.get(idx) );
	time1 = toc();
	tic();
	out_image2 = MRF( prob_map, Xv, Yv );
	time2 = toc();
	tic();
	[r,c,~] = size(prob_map);
	gt = zeros(r,c);
	gt( sub2ind([r,c], [Xt(:,1);Xv(:,1)], [Xt(:,2);Xv(:,2)]) ) = [Yt;Yv];   
	out_image3 = CRF( prob_map, gt );
	time3 = toc();

	out_maps = {out_image1,out_image2,out_image3};
	time_taken = [time1,time1+time2,time1+time3];
end

function map = CRF( img, gt_img )
	addpath(genpath('libs/JustinsGraphicalModelsToolboxPublic'));
	N     = 1;  % number of training images
	[siz_r,siz_c,siz_b]   = size(img); % size of training images
	rho   = .5; % TRW edge appearance probability
	nvals = max(gt_img(:));  % this problem is binary
	model = gridmodel(siz_r,siz_c,nvals);
	feats{1} = reshape(img,[siz_r*siz_c,siz_b]);
	labels{1} = gt_img(:);
	efeats = [];
	loss_spec = 'trunc_cl_trw_5';
	crf_type  = 'linear_linear';
	options.derivative_check = 'off';
	options.rho         = rho;
	options.print_times = 1;
	options.nvals       = nvals;	

	p = train_crf(feats,efeats,labels,model,loss_spec,crf_type,options);

	[b_i b_ij] = eval_crf(p,feats{1},efeats,model,loss_spec,crf_type,rho);
	b_i = reshape(b_i',[siz_r siz_c nvals]);
	[~,map] = max(b_i,[],3);
	rmpath(genpath('libs/JustinsGraphicalModelsToolboxPublic'));
end 



function [image, groundTruth, labelNames] = load_dataset(dataset_name)
    if strcmp(dataset_name, 'IP')
	    groundTruth = load( 'Indian_pines_gt.mat' );
	    image = load( 'Indian_pines_corrected.mat' );
	    labelNames = importdata( 'IP_classes.txt' );
	    groundTruth = groundTruth.indian_pines_gt;
	    image = image.indian_pines_corrected;
    elseif strcmp(dataset_name, 'PaviaU')
	    groundTruth = load( 'PaviaU_gt.mat' );
	    image = load( 'PaviaU.mat' );
	    labelNames = importdata( 'PaviaU_classes.txt' );
	    groundTruth = groundTruth.paviaU_gt;
	    image = image.paviaU;
    elseif strcmp(dataset_name, 'PaviaC')
	    groundTruth = load( 'Pavia_gt.mat' );
	    image = load( 'Pavia.mat' );
	    labelNames = importdata( 'PaviaC_classes.txt' );
	    groundTruth = groundTruth.pavia_gt;
	    image = image.pavia;
    elseif strcmp(dataset_name, 'Salinas')
	    groundTruth = load( 'Salinas_gt.mat' );
	    image = load( 'Salinas_corrected.mat' );
	    labelNames = importdata( 'Salinas_classes.txt' );
	    groundTruth = groundTruth.salinas_gt;
	    image = image.salinas_corrected;
    else
        error('Incorrect dataset name. Available datasets are: IP, PaviaU, PaviaC, Salinas.');
    end
    image = double(image);
end
function [] = make()
    build_mex = true; % builds mex (C++ compiler needed) 
                      % required for SVM
                      % faster MRF and CRF
                      
    %install datasets
    disp('Downloading datasets:');
    %install_datasets();
    %install classifiers
    disp('Downloading and installing classifiers:');
    %install_classifiers(build_mex);
    %install libs
    disp('Downloading and installing libraries:');
    install_libs(build_mex);

end



function [] = install_classifiers(build_mex)
    %Downloads and install liblinear, libsvm, GPML from
    %
    urls = {'https://github.com/cjlin1/liblinear/archive/master.zip',...
            'https://github.com/cjlin1/libsvm/archive/master.zip',...
            'http://gaussianprocess.org/gpml/code/matlab/release/gpml-matlab-v4.1-2017-10-19.zip'};
    
    output_folder = './classifiers/';
    out_filenames = {'liblinear.zip','libsvm.zip','gpml.zip'};
    download(urls, output_folder, out_filenames);
    
    for i = 1:length(out_filenames)
        unzip([output_folder, out_filenames{i}],output_folder);    
    end
    

    
    if nargin > 0
        if build_mex
            run([output_folder, '/libsvm-master/matlab/make.m']);
            run([output_folder, '/liblinear-master/matlab/make.m']);
        end
    end
end

function [] = install_libs(build_mex)
    %Downloads and installs libs from 
    %https://www.cs.ubc.ca/~schmidtm/Software/UGM.html
    %https://people.cs.umass.edu/~domke/JGMT/
    %http://www.vlfeat.org/download.html
    %http://vision.csd.uwo.ca/code/
    %http://people.csail.mit.edu/mrub/
    urls = {'https://www.cs.ubc.ca/~schmidtm/Software/UGM_2011.zip',...
            'https://people.cs.umass.edu/~domke/JGMT/JGMT4.zip',...
            'http://www.vlfeat.org/download/vlfeat-0.9.21-bin.tar.gz',...
            'http://vision.csd.uwo.ca/code/maxflow-v3.01.zip',...
            'http://people.csail.mit.edu/mrub/code/maxflow-1.1.zip'};
        
    output_folder = './libs/';
    download(urls, output_folder);
    
    %unzip files
    zip_files = {'UGM_2011.zip', 'JGMT4.zip', ...
                 'vlfeat-0.9.21-bin.tar.gz','maxflow-1.1.zip',...
                 'maxflow-v3.01.zip'};
    lib_folders = {'','','','maxflow','maxflow/maxflow-v3.0'};
    unzip_fun={@unzip,@unzip,@untar,@unzip,@unzip};
    for i = 1:length(zip_files)
        unzip_fun{i}([output_folder, zip_files{i}], ...
                     [output_folder, lib_folders{i}]);
    end
    
    
    if nargin > 0
        if build_mex
            run([output_folder, 'UGM/mexAll.m']);
            run([output_folder, ...
                'JustinsGraphicalModelsToolboxPublic/compile.m']);
            run([output_folder, 'maxflow/make.m']);
            
            changefile = [output_folder, 'UGM/decode/UGM_Decode_GraphCut.m'];
            fptr = fopen( changefile, 'r');
            lines = {};
            while ~feof(fptr)
                aline = fgetl(fptr);
                if strcmp(aline,'if edgeStruct.useMex && exist(''maxflowmex'')==3 % Use mex interface to maxflow code')
                    lines = [lines, 'if exist(''maxflowmex'')==3 % Use mex interface to maxflow code'];
                else
                    lines = [lines, aline];
                end
            end
            fclose(fptr);
            fptr = fopen(changefile, 'w');
            for i = 1:length(lines)
                fprintf(fptr, '%s\n', lines{i});
            end
            fclose(fptr);
            
            
        end
    end
end

function [] = install_datasets()
    %Downloads benchmark images and ground truth from 
    %http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
    urls = {'http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',...
            'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat',...
            'http://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat',...
            'http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat',...
            'http://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat',...
            'http://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat',...
            'http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',...
            'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat'};
    output_folder = './datasets/';
    download(urls, output_folder);
end

function [] = download(urls, output_folder, out_filenames)
    num_urls = length(urls);
    reverseStr = '';
    options = weboptions('Timeout',Inf);
    for i = 1:num_urls
        filename = strsplit(urls{i},'/');
        
        msg = sprintf('Downloading %s....(%d/%d files done!)',...
              filename{end},i-1, num_urls); 
        fprintf([reverseStr, msg]);
        reverseStr = repmat(sprintf('\b'), 1, length(msg));
        if nargin > 2
            websave([output_folder,out_filenames{i}],urls{i},options);
        else
            websave([output_folder,filename{end}],urls{i},options);
        end
    end
    fprintf(reverseStr);
end

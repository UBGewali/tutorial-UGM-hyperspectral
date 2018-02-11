%startup.m
clear all;
rng('shuffle');
addpath(genpath('datasets'));
addpath(genpath('features'));
addpath(genpath('models'));

addpath('libs');
addpath(genpath('libs/UGM'));
addpath(genpath('libs/maxflow'));
%addpath(genpath('libs/JustinsGraphicalModelsToolboxPublic')); 
%JGMT has functions that have name conflicts with MATLAB inbuilt functions.
%So JGMT is added only in CRF.m
%vlfeat
run('./libs/vlfeat-0.9.21/toolbox/vl_setup.m');

addpath('classifiers');
addpath(genpath('classifiers/libsvm-master'));
addpath(genpath('classifiers/liblinear-master'));
%gpml
run('./classifiers/gpml-matlab-v4.1-2017-10-19/startup.m');

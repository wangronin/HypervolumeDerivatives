%%%-----------------------V1.3.5 17/07/2024-----------------------------%%%

%------------------------------------------%
clearvars
close all
%------------------------------------------%

%--------RSG PARAMETERS--------------------%
%--DisplayPlots--%
show_plots = false;
save_plots = false;
%--Size of interpolation--%
interpol_size = 5000; 
%--Final Reference Size--%
ref_size = 300;

%--Cleaning--%
clean_method = 'long';
threshold = 0.6;
% clean_method = 'cond';
% threshold = 150;
% clean_method = 'off';
% threshold = 'off';
% clean_method = 'area';
% threshold = 1.5;

% %SPECIAL CASE:
% threshold = 'zero';
% threshold = 'decreasing';

%COMPONENT DETECTION PARAMETERS
% epsInterval = [0.25,0.30]; %TESTS!!! MODIFIED VALUE
% eps_def = 0.05; 
% minptsInterval = [4,4];
epsInterval = [10.2,10.2]; %TESTS!!! MODIFIED VALUE
eps_def = 0.05;
minptsInterval = [2,2];

%TEST PROBLEM
problem_name = 'DTLZ2_SMS-EMOA';
%------------------------------------------%

%--------GENERATE STARTING POINTS----------%
% Py = SampleOptimum(problem_name,sample_size);

% load('DTLZ1_Ay_outliers.mat')
% Py = Ay;

% Py = readmatrix('F:\Postdoc\HaoReferences\ZDT\ZDT3_NSGA-III_run_6_lastpopu_y_gen300.csv');

% Py = readmatrix('F:\Postdoc\SendHao\AllPaperExamples\DTLZ2_NSGA-II_run_1_lastpopu_y_gen300.csv');

load('DTLZ2-DpNpaperexample.mat')
Py(11,:) = [1,0,0];
Py(13,:) = [0,1,0];
Py(68,:) = [0,0,1];
%------------------------------------------%

%--------SAVE ALGORITHM PARAMETER----------%
save([pwd '/Plots/' problem_name '_RSG_parameters.mat']);
%------------------------------------------%


%----------PLOT STARTING POINTS------------%
if size(Py,2) == 2
    scatter(Py(:,1),Py(:,2),'.')
else
    scatter3(Py(:,1),Py(:,2),Py(:,3),'.')
end
%------------------------------------------%


%-------------------RSG--------------------%
[Z_means,Z_medoids] = RSG( Py, interpol_size, ref_size, clean_method, threshold, problem_name, epsInterval, eps_def, minptsInterval,show_plots,save_plots);
%------------------------------------------%

%---------------Save Result----------------%
save([pwd '/Data/' problem_name '_ref.mat'],'Z_medoids','Z_means')
%------------------------------------------%
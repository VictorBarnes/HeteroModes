% Combine output of simulateFC_optimise.sh into one mat file
clear
clc

% Setup project by loading necessary functions
setupProject

% Load config file
config = jsondecode(fileread("config.json"));

heteroLabel = "SAaxis";
dataDir = fullfile(config.project_dir, "data");
resultsDir = fullfile(config.project_dir, "results");

% Load valid cs parameter combinations
csParamCombs_valid = readmatrix(fullfile(dataDir, sprintf("hetero-%s_csParamCombs_valid.csv", ...
    heteroLabel)));
nCombs = size(csParamCombs_valid, 1);

edgeFCcorr = nan(nCombs);
nodeFCcorr = nan(nCombs);
FCDks = nan(nCombs);
for ii=1:nCombs
    % Load data
    alpha = csParamCombs_valid(ii, 1);
    beta = csParamCombs_valid(ii, 2);
    filename = sprintf("hetero-%s_alpha-%.1f_beta-%.1f_empDset-hcp_nRuns-50_nSubj-50_crossVal-False_simulateFCresults.mat", ...
        heteroLabel, alpha, beta);
    
    try
        results = load(fullfile(resultsDir, "simulateFC", "optimise", "temp", filename));
    catch exception
        fprintf("File not found for alpha: %.1f, beta: %.1f\n", alpha, beta)
        continue
    end
    edgeFCcorr(ii) = results.edgeFCcorr;
    nodeFCcorr(ii) = results.nodeFCcorr;
    FCDks(ii) = results.FCDks;
end

outputFolder = fullfile(config.project_dir, 'results', 'simulateFC', 'optimise');
outputDesc = "hetero-%s_empDset-hcp_nRuns-%i_nSubj-50_crossVal-False_simulateFCresults.mat";
save(fullfile(outputFolder, sprintf(outputDesc, heteroLabel, 50)), 'csParamCombs_valid', ...
    'edgeFCcorr', 'nodeFCcorr', 'FCDks');

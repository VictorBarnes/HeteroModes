% Combine output of simulateFC_optimise.sh into one mat file
clear
clc

% Setup project by loading necessary functions
setupProject

% Load config file
config = jsondecode(fileread("config.json"));

heteroLabel = "myelinmap";
dataDir = fullfile(config.project_dir, "data");
resultsDir = fullfile(config.project_dir, "results");

% Load cortex mask
medialMask = dlmread(sprintf('%s/atlas-%s_space-%s_den-%s_hemi-%s_medialMask.txt', config.surface_dir, ...
    config.atlas, config.space, config.den, config.hemi));
cortexInds = find(medialMask);

% Load valid cs parameter combinations
csParamCombs_valid = readmatrix(fullfile(dataDir, sprintf("hetero-%s_csParamCombs_finePositive_valid.csv", ...
    heteroLabel)));
nCombs = size(csParamCombs_valid, 1);

edgeFCcorr = nan(nCombs);
nodeFCcorr = nan(nCombs);
FCDks = nan(nCombs);
csMin = nan(nCombs, 1);
csMax = nan(nCombs, 1);
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

    % Load cs maps to save min and max values
    desc = 'hetero-%s_atlas-%s_space-%s_den-%s_surf-%s_hemi-%s_n-%i_alpha-%.1f_beta-%.1f_maskMed-True';
    csMap = readmatrix(fullfile(config.emode_dir, 'cmaps', sprintf(desc, heteroLabel, config.atlas, ...
        config.space, config.den, config.surf, config.hemi, config.n_modes, alpha, beta) + "_cmap.txt")); 

    % Need to take sqrt because these values have been squared according to the NFT wave equation
    csMin(ii) = min(sqrt(csMap(cortexInds)));
    csMax(ii) = max(sqrt(csMap(cortexInds)));
end

outputFolder = fullfile(config.project_dir, 'results', 'simulateFC', 'optimise');
outputDesc = "hetero-%s_empDset-hcp_nRuns-%i_nSubj-50_crossVal-False_simulateFCresults_finePositive.mat";
save(fullfile(outputFolder, sprintf(outputDesc, heteroLabel, 50)), 'csParamCombs_valid', ...
    'edgeFCcorr', 'nodeFCcorr', 'FCDks', 'csMin', 'csMax');

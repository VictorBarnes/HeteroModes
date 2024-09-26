%% Plot results from simulateFC_optimise.m
clear
clc

% Setup project by loading necessary functions
setupProject

% Load surface and mode parameters
config = jsondecode(fileread(fullfile(pwd, "config.json")));
projDir = config.project_dir;
resultsDir = config.results_dir;
heteroLabel = 'myelinmap';

% Load homogeneous mode results
desc = "hetero-%s_empDset-hcp_nRuns-50_nSubj-50_crossVal-False_simulateFCresults.mat";
homoResults = load(fullfile(projDir, "results", "simulateFC", "optimise", sprintf(desc, "None")));

% Load heterogeneous mode results
desc = "hetero-%s_empDset-hcp_nRuns-50_nSubj-50_crossVal-False_simulateFCresults_finePositive.mat";
heteroResults = load(fullfile(projDir, "results", "simulateFC", "optimise",...
    sprintf(desc, heteroLabel)));
alphaVals = unique(heteroResults.csParamCombs_valid(:, 1));
alphaVals(alphaVals < -1.4 | alphaVals > 2.8) = [];
betaVals = unique(heteroResults.csParamCombs_valid(:, 2));

% Reshape results into nAlpha x nBeta matrices
edgeFC_reshaped = nan(length(alphaVals), length(betaVals));
nodeFC_reshaped = nan(length(alphaVals), length(betaVals));
FCDks_reshaped = nan(length(alphaVals), length(betaVals));
csMin_reshaped = nan(length(alphaVals), length(betaVals));
csMax_reshaped = nan(length(alphaVals), length(betaVals));
combinedMetrics = nan(length(alphaVals), length(betaVals));
for ii = 1:length(heteroResults.csParamCombs_valid)   
    % Get alpha and beta indices
    alphaInd = find(alphaVals == heteroResults.csParamCombs_valid(ii, 1));
    betaInd = find(betaVals == heteroResults.csParamCombs_valid(ii, 2));

    % Get evaluation metrics for each comb
    edgeFC_reshaped(alphaInd, betaInd) = heteroResults.edgeFCcorr(ii);
    nodeFC_reshaped(alphaInd, betaInd) = heteroResults.nodeFCcorr(ii);
    FCDks_reshaped(alphaInd, betaInd) = heteroResults.FCDks(ii);
    combinedMetrics(alphaInd, betaInd) = heteroResults.edgeFCcorr(ii) ...
        + heteroResults.nodeFCcorr(ii) + (1 - heteroResults.FCDks(ii));

    % Get cs min and max values for each comb and convert to m/s
    csMin_reshaped(alphaInd, betaInd) = heteroResults.csMin(ii)/1000;
    csMax_reshaped(alphaInd, betaInd) = heteroResults.csMax(ii)/1000;
end

maskPhysRange = true;
saveFile = "hetero-%s_empDset-hcp_nRuns-50_nSubj-50_crossVal-False_simulateFCresults";
% Mask values that are outside the physiological range
if maskPhysRange
    % Get min and max mask and combine them
    maskMin = find(csMin_reshaped <= 0.1);
    maskMax = find(csMax_reshaped >= 150);
    maskCombined = union(maskMin, maskMax);
    
    % Apply mask
    edgeFC_reshaped(maskCombined) = NaN;
    nodeFC_reshaped(maskCombined) = NaN;
    FCDks_reshaped(maskCombined) = NaN;
    combinedMetrics(maskCombined) = NaN;
    
    % Update save file
    saveFile = saveFile + "_masked";
end

% Convert alphaVals and betaVals to cell arrays for plotting
alphaVals_cell = cellfun(@num2str, num2cell(alphaVals), 'UniformOutput', false);
betaVals_cell = cellfun(@num2str, num2cell(betaVals), 'UniformOutput', false);

%% Plot matrices
% tl = tiledlayout(3, 1, 'TileSpacing', 'tight');
% title(tl, sprintf('Optimisation Results (Hetero: %s)', heteroLabel));
fsize = 14;
noAxisTicks = false;

%% Plot hetero edge FC - homo edge FC
figure('Position', [100, 100, 984, 805], 'Visible', 'on')
data = edgeFC_reshaped - homoResults.edgeFCcorr;
h1 = heatmap(betaVals_cell, alphaVals_cell, data, 'FontSize', fsize); 
% h1.Title = 'Edge FC correlation difference';
h1.XLabel = '\beta'; h1.YLabel = '\alpha';
h1.Colormap = bluewhitered_mg('clims', [min(data, [], 'all'), max(data, [], 'all')], 'autoscaleColors', true);
h1.CellLabelFormat = '%.2f';
annotation('textarrow',[0.98,1.0],[0.5,0.5],'string', 'Heterogeneous - Homogeneous', ...
      'HeadStyle','none','LineStyle','none','HorizontalAlignment','center','TextRotation',90, 'FontSize', 16);
if noAxisTicks
    ax = gca;
    ax.XDisplayLabels = nan(size(ax.XDisplayData));
    ax.YDisplayLabels = nan(size(ax.YDisplayData));
end

%% Plot hetero node FC - homo node FC
figure('Position', [100, 100, 984, 805], 'Visible', 'on')
data = nodeFC_reshaped - homoResults.nodeFCcorr;
h2 = heatmap(betaVals_cell, alphaVals_cell, data, 'FontSize', fsize); 
% h2.Title = {'Node FC correlation'; '(heterogeneous - homogeneous)'};
h2.XLabel = '\beta'; h2.YLabel = '\alpha';
h2.Colormap = bluewhitered_mg('clims', [min(data, [], 'all'), max(data, [], 'all')], 'autoscaleColors', true);
h2.CellLabelFormat = '%.2f';
annotation('textarrow',[0.98,1.0],[0.5,0.5],'string', 'Heterogeneous - Homogeneous', ...
      'HeadStyle','none','LineStyle','none','HorizontalAlignment','center','TextRotation',90, 'FontSize', 16);

%% Plot KS of hetero FCD - KS of homo FCD
figure('Position', [100, 100, 984, 805], 'Visible', 'on')
data = (1 - FCDks_reshaped) - (1- homoResults.FCDks);
h3 = heatmap(betaVals_cell, alphaVals_cell, data, 'FontSize', fsize); 
% h3.Title = {'(1 - FCD KS statistic)'; '(heterogeneous - homogeneous)'};
h3.XLabel = '\beta'; h3.YLabel = '\alpha';
h3.Colormap = bluewhitered_mg('clims', [min(data, [], 'all'), max(data, [], 'all')], 'autoscaleColors', true);
h3.CellLabelFormat = '%.2f';
annotation('textarrow',[0.98,1.0],[0.5,0.5],'string', 'Heterogeneous - Homogeneous', ...
      'HeadStyle','none','LineStyle','none','HorizontalAlignment','center','TextRotation',90, 'FontSize', 16);


%% Combine evaluation metrics into one plot
figure('Position', [132 27 1300 1050]);
nexttile
data = combinedMetrics - (homoResults.edgeFCcorr + homoResults.nodeFCcorr + (1 - homoResults.FCDks));
h4 = heatmap(betaVals_cell, alphaVals_cell, data, 'Fontsize', 14);
% h4.Title = {'Edge FC + Node FC - FCD'; '(heterogeneous - homogeneous)'};
h4.XLabel = '\beta'; h4.YLabel = '\alpha';
h4.Colormap = bluewhitered_mg('clims', [min(data, [], 'all'), max(data, [], 'all')], 'autoscaleColors', true);
h4.CellLabelFormat = '%.2f';
annotation('textarrow',[0.9,1.0],[0.5,0.5],'string', 'Heterogeneous - Homogeneous', ...
      'HeadStyle','none','LineStyle','none','HorizontalAlignment','center','TextRotation',90, 'FontSize', 16);
% Save results
% savecf(fullfile(resultsDir, 'simulateFC', 'optimise', sprintf(saveFile, heteroLabel)), ".png", 150)

%% Plot cs min and max values
figure('Position', [100, 100, 2500, 1000], 'Visible', 'off')
tl = tiledlayout(1, 2);
title(tl, sprintf('cs values (Hetero: %s)', heteroLabel));
cmap = viridis;

% Convert alphaVals and betaVals to cell arrays for plotting
alphaVals_cell = cellfun(@num2str, num2cell(alphaVals), 'UniformOutput', false);
betaVals_cell = cellfun(@num2str, num2cell(betaVals), 'UniformOutput', false);

% Plot min cs values (in m/s)
nexttile
h1 = heatmap(betaVals_cell, alphaVals_cell, csMin_reshaped); 
h1.Title = 'Minimum cs value (m/s)';
h1.XLabel = 'beta'; h1.YLabel = 'alpha'; 
h1.Colormap = cmap;
h1.CellLabelFormat = '%.3f';

% Plot max cs values (in m/s)
nexttile
h1 = heatmap(betaVals_cell, alphaVals_cell, csMax_reshaped, ...
    'ColorScaling', 'log');  % 'ColorLimits', [min(csMax_reshaped, [], "all"), 150]
h1.Title = 'Maximum cs value (m/s)';
h1.XLabel = 'beta'; h1.YLabel = 'alpha'; 
h1.Colormap = cmap;
h1.CellLabelFormat = '%.2g';

% Save results
savecf(fullfile(resultsDir, 'simulateFC', 'optimise', ...
    sprintf("hetero-%s_cmapRange", heteroLabel)), ".png", 100)
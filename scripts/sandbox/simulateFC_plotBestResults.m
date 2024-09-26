%% Plot BEST results from simulateFC_optimise.m
clear
clc

% Setup project by loading necessary functions
setupProject
addpath(genpath("/fs03/kg98/vbarnes/temp/BrainEigenmodes/functions_matlab/cbrewer/"))

% Load surface and mode parameters
config = jsondecode(fileread(fullfile(pwd, "config.json")));
projDir = config.project_dir;
resultsDir = config.results_dir;
heteroLabel = 'myelinmap';
alpha = 2.4;
beta = 1.0;
nRuns = 50;
nParcels = 180;

% Load homogeneous mode results
desc = 'hetero-%s_alpha-%.1f_beta-%.1f_empDset-hcp_nRuns-%i_nSubj-50_crossVal-False_simulateFCresults_saveAll.mat';
homoResults = load(fullfile(projDir, "results", "simulateFC", "optimise", sprintf(desc, "None", 0, 0, nRuns)));
heteroResults = load(fullfile(projDir, "results", "simulateFC", "optimise", sprintf(desc, heteroLabel, alpha, beta, nRuns)));

%% Plot figure
fig = figure('Position', [100, 500, 1800, 400]);
tl = tiledlayout(1, 4, 'TileSpacing', 'loose');
triuInds = find(triu(ones(nParcels, nParcels), 1));
fontsize_axis = 14;
fontsize_label = 16;
resultsCurr = heteroResults;

%%% Plot FC
ax1 = nexttile;
extraDiagonal = 7;
FC_combined_model = zeros(nParcels + extraDiagonal);
FC_combined_model(find(triu(ones(nParcels + extraDiagonal),1 + extraDiagonal))) = resultsCurr.simFC_avg(triuInds);
FC_combined_emp = zeros(nParcels + extraDiagonal);
FC_combined_emp(find(triu(ones(nParcels + extraDiagonal),1 + extraDiagonal))) = resultsCurr.empFC_avg(triuInds);

FC_combined = FC_combined_model + FC_combined_emp';

pointStart = 1-0.5; pointEnd = nParcels + extraDiagonal + 0.5;

imagesc(FC_combined)
hold on;
plot([pointStart+1+extraDiagonal pointEnd], pointStart*ones(1,2), 'k-', 'linewidth', 1)
plot(pointEnd*ones(1,2), [pointStart pointEnd-1-extraDiagonal], 'k-', 'linewidth', 1)
plot([pointStart pointEnd-1-extraDiagonal], pointEnd*ones(1,2), 'k-', 'linewidth', 1)
plot(pointStart*ones(1,2), [pointStart+1+extraDiagonal pointEnd], 'k-', 'linewidth', 1)
for ii=1:nParcels-1
    plot((pointStart+1+extraDiagonal+(ii-1))*ones(1,2), pointStart+[0 1]+(ii-1), 'k-', 'linewidth', 1)
    plot(pointStart+1+extraDiagonal+[0 1]+(ii-1), (pointStart+1+(ii-1))*ones(1,2), 'k-', 'linewidth', 1)

    plot((pointStart+1+(ii-1))*ones(1,2), pointStart+1+extraDiagonal+[0 1]+(ii-1), 'k-', 'linewidth', 1)
    plot(pointStart+[0 1]+(ii-1), (pointStart+1+extraDiagonal+(ii-1))*ones(1,2), 'k-', 'linewidth', 1)
end
colormap(ax1, bluewhitered)
set(gca, 'fontsize', fontsize_axis, 'ticklength', [0.02 0.02])
axis off
hold off
text(-5, (nParcels + extraDiagonal)/2, 'Empirical FC', 'rotation', 90, 'FontSize', fontsize_axis, 'horizontalalignment', 'center', 'verticalalignment', 'bottom')
text((nParcels + extraDiagonal)/2, -2, 'Model FC', 'FontSize', fontsize_axis, 'horizontalalignment', 'center', 'verticalalignment', 'bottom')

%%% Plot Edge FC
ax2 = nexttile;
hold on
plot(resultsCurr.simFC_avg, resultsCurr.empFC_avg, 'k.', 'markersize', 4)
plot(resultsCurr.simFC_avg, polyval(polyfit(resultsCurr.simFC_avg, resultsCurr.empFC_avg, 1), resultsCurr.simFC_avg), 'r-', 'linewidth', 2)
xlim([min(resultsCurr.simFC_avg(triuInds)), max(resultsCurr.simFC_avg(triuInds))])
hold off
set(ax2, 'fontsize', fontsize_axis, 'ticklength', [0.02, 0.02], 'LineWidth', 1)
xlabel('Model', 'FontSize', fontsize_axis)
ylabel('Empirical', 'FontSize', fontsize_axis)
title('Edge FC', 'FontSize', fontsize_label, 'FontWeight', 'normal')
[rho, ~] = corr(resultsCurr.simFC_avg(triuInds), resultsCurr.empFC_avg(triuInds), 'type', 'pearson');
text(max(get(ax2,'xlim')), min(get(ax2,'ylim'))+0.1*[max(get(ax2,'ylim'))-min(get(ax2,'ylim'))], sprintf('r = %.2f', rho), 'color', 'k', ...
    'FontSize', fontsize_axis, 'verticalalignment', 'middle', 'horizontalalignment', 'right');
box off

%%% Plot Node FC
ax3 = nexttile;
simNodeFC = mean(resultsCurr.simFC_avg - diag(diag(resultsCurr.simFC_avg)), 2);
empNodeFC = mean(resultsCurr.empFC_avg - diag(diag(resultsCurr.empFC_avg)), 2);
hold on;
plot(simNodeFC, empNodeFC, 'k.', 'markersize', 10)
plot(simNodeFC, polyval(polyfit(simNodeFC,empNodeFC,1), simNodeFC), ...
    'r-', 'linewidth', 2);
hold off;
set(ax3, 'FontSize', fontsize_axis, 'ticklength', [0.02, 0.02], 'LineWidth', 1)
xlabel('Model', 'FontSize', fontsize_axis)
ylabel('Empirical', 'FontSize', fontsize_axis)
title('Node FC', 'FontSize', fontsize_label, 'fontweight', 'normal')
[rho,pval] = corr(simNodeFC, empNodeFC, 'type', 'pearson');
text(max(get(gca,'xlim')), min(get(gca,'ylim'))+0.1*[max(get(gca,'ylim'))-min(get(gca,'ylim'))], sprintf('r = %.2f', rho), 'color', 'k', ...
    'fontsize', fontsize_axis, 'verticalalignment', 'middle', 'horizontalalignment', 'right');
box off

%%% Plot phase FCD
cb = cbrewer('qual', 'Set3', 12, 'pchip');
ax4 = nexttile;
[~, ~, FCDks] = kstest2(resultsCurr.empFCDs(:), resultsCurr.simFCDs(:));
hold on;
histogram(resultsCurr.empFCDs(:), 'normalization', 'pdf', 'facecolor', cb(1,:), 'edgecolor', 'none', 'displayname', 'data', 'facealpha', 0.6)
histogram(resultsCurr.simFCDs(:), 'normalization', 'pdf', 'facecolor', cb(4,:), 'edgecolor', 'none', 'displayname', 'model', 'facealpha', 0.6)
hold off;
objects = findobj(gca, 'type', 'histogram');
set(ax4, 'fontsize', fontsize_axis, 'ticklength', [0.02, 0.02], 'xlim', [0 1], 'children', flipud(get(gca,'children')), 'LineWidth', 1)
leg = legend([objects(2) objects(1)], {'Empirical', 'Model'}, 'fontsize', fontsize_axis, 'location', 'northeast', 'interpreter', 'none', 'box', 'off', 'numcolumns', 1);
leg.ItemTokenSize = leg.ItemTokenSize/2;
leg_names = get(leg, 'string');
xlabel('Synchrony', 'fontsize', fontsize_axis)
ylabel('pdf', 'fontsize', fontsize_axis)
title('FCD', 'fontsize', fontsize_label, 'fontweight', 'normal')
text(max(get(gca,'xlim')), min(get(gca,'ylim'))+0.1*[max(get(gca,'ylim'))-min(get(gca,'ylim'))], sprintf('KS = %.2f', FCDks), 'color', 'k', ...
    'fontsize', fontsize_axis, 'verticalalignment', 'middle', 'horizontalalignment', 'right');
box off

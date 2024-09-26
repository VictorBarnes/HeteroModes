%% Code to plot heterogeneous modes
% TODO: change geometric to homogeneous
clear
clc

% Setup project by loading necessary functions
setupProject

% Load config file
config = jsondecode(fileread("config.json"));
atlas = config.atlas;
space = config.space;
den = config.den;
surf = config.surf;
hemi = config.hemi;
nModes = config.n_modes;
realHeteroMaps = config.hetero_maps;
emodeDir = config.emode_dir;
surfDir = config.surface_dir;
projDir = '/fs04/kg98/vbarnes/HeteroModes';

% Set parameters for heterogeneous modes
modeParams_default = struct('heteroLabel', 'myelinmap', 'alpha', 1.0, 'beta', 1.0);
modeParams = [struct('alpha', 2.4, 'beta', 1.0)];
nHeteroBSs = length(modeParams);     % Number of heterogeneous basis sets

% Load Yeo surface file
[vertices, faces] = read_vtk(sprintf('%s/atlas-%s_space-%s_den-%s_surf-%s_hemi-%s_surface.vtk', ...
    surfDir, atlas, space, den, surf, hemi));
surface.vertices = vertices';
surface.faces = faces';
% Load cortex mask
medialMask = dlmread(sprintf('%s/atlas-%s_space-%s_den-%s_hemi-%s_medialMask.txt', surfDir, atlas, ...
    space, den, hemi));
cortexInds = find(medialMask);

% Load homogeneous eigenmodes and eigenvalues
desc = 'hetero-%s_atlas-%s_space-%s_den-%s_surf-%s_hemi-%s_n-%i_alpha-%.1f_beta-%.1f_maskMed-True';
homoModes = dlmread(fullfile(emodeDir, sprintf(desc, "None", atlas, space, den, surf, hemi, ...
    nModes, 0.0, 0.0) + "_emodes.txt"));
homoEvals = dlmread(fullfile(emodeDir, sprintf(desc, "None", atlas, space, den, surf, hemi, ...
    nModes, 0.0, 0.0) + "_evals.txt"));

% Load heterogeneous eigenmodes and eigenvalues
cmaps = zeros(size(homoModes, 1), nHeteroBSs);
heteroModes = zeros([size(homoModes), nHeteroBSs]);
heteroEvals = zeros(nHeteroBSs, nModes);
for ii=1:nHeteroBSs
% Extract current parameter values from the struct
    currentParams = modeParams_default;
    paramNames = fieldnames(modeParams_default);
    
    % Set default values for parameters not specified
    for jj=1:length(paramNames)
        if isfield(modeParams(ii), paramNames{jj})
            currentParams.(paramNames{jj}) = modeParams(ii).(paramNames{jj});
        end
    end
    
    % Load propagation maps
    cmaps(:, ii) = dlmread(fullfile(emodeDir, 'cmaps', sprintf(desc, currentParams.heteroLabel, atlas, space, ...
        den, surf, hemi, nModes, currentParams.alpha, currentParams.beta) + "_cmap.txt"));  

    % Load hetero modes and evals
    modes_current = dlmread(fullfile(emodeDir, sprintf(desc, currentParams.heteroLabel, atlas, ...
        space, den, surf, hemi, nModes, currentParams.alpha, currentParams.beta) + "_emodes.txt"));
    heteroModes(:, :, ii) = modes_current(:, 1:nModes);
    evals_current = dlmread(fullfile(emodeDir, sprintf(desc, currentParams.heteroLabel, atlas, space, ...
        den, surf, hemi, nModes, currentParams.alpha, currentParams.beta) + "_evals.txt")); 
    heteroEvals(ii, :) = evals_current(1:nModes);
end

% Find min and max values in cmaps for plotting on using same x axis
cmapLimsGlobal = [round(min(cmaps(cortexInds, :), [], "all")), round(max(cmaps(cortexInds, :), [], "all"))];

% Set reordering parameters
reorder = false;
withinGroups = true;
% Only save with 'withinGroups' tag in filename if `reorder` is falsealphaVals
if reorder
    reorderText = sprintf("1_withinGroups-%i", withinGroups);
else
    reorderText = num2str(reorder);
end

%% Plot heterogeneous modes
% TODO: change this to plot the cmaps and distributions first then modes in one plotBrain call. This
% should make the plot look nicer and 

nModes_vis = 200; % number of modes to visualise in correlation matrix
modesToPlot = [2, 3, 4, 10, 50, 100];
% modesToPlot = [10, 14];
nModesToPlot = length(modesToPlot);
plotHist = 0;   % boolean for whether to plot cmap distribution or not
plotCorr = 1;   % boolean for whether to plot correlation matrix
plotArrangement = 'vertical';

%%% PLOT HOMOGENEOUS MODES
figure('Position', [100, 100 ,1400, 300]);
tl1 = tiledlayout('flow', 'TileSpacing','tight');
[~, ~, tl2, tl3] = plotBrain('lh', {surface, medialMask, homoModes(:, modesToPlot)}, 'parent', tl1, ...
    'groupBy', 'data', 'colormap', @bluewhitered, 'tiledLayoutOptions', {'flow', 'TileSpacing', 'tight'}, ...
    'view', {'ll', 'lm'}, 'tiledLayout2Options', {1, 2, 'TileSpacing', 'none'});
tl2.Layout.Tile = 1;
tl2.Layout.TileSpan = [1, nModesToPlot];    % TODO
title(tl2, 'Homogeneous modes', 'FontWeight', 'bold');
% Plot mode labels
for ii=1:length(modesToPlot)
    switch plotArrangement
        case 'horizontal'
            xlab = xlabel(tl3{ii}, sprintf('mode %i', modesToPlot(ii)), 'FontSize', 10, 'Rotation', 0, ...
                'VerticalAlignment', 'middle');
        case 'vertical'
            ylab = ylabel(tl3{ii}, {"mode"; modesToPlot(ii)}, 'FontSize', 10, 'Rotation', 0, ...
                'VerticalAlignment', 'middle');
    end
end

%%% PLOT HETEROGENEOUS MODES
% If a hetero mode is the same as the corresponding geometric mode but just flipped then flip 
% the sign of the hetero mode for better visual comparison
corrs_diag = diag(corr(homoModes, heteroModes)); 
mask = corrs_diag < -0.8;
heteroModes(:, mask) = heteroModes(:, mask) * -1;
[~, ~, tl2, tl3] = plotBrain('lh', {surface, medialMask, heteroModes(:, modesToPlot)}, 'parent', tl1, ...
    'groupBy', 'data', 'colormap', @bluewhitered, 'tiledLayoutOptions', {'flow', 'TileSpacing', 'tight'}, ...
    'view', {'ll', 'lm'}, 'tiledLayout2Options', {1, 2, 'TileSpacing', 'none'});
tl2.Layout.Tile = 2;
tl2.Layout.TileSpan = [1, nModesToPlot];    % TODO
title(tl2, 'Heterogeneous modes (T1w/T2w)', 'FontWeight', 'bold')     % , sprintf('%s | \\alpha: %.1f, \\beta: %.1f', 'T1w/T2w', modeParams.alpha, modeParams.beta)
% Plot mode labels
for ii=1:length(modesToPlot)
    switch plotArrangement
        case 'horizontal'
            xlab = xlabel(tl3{ii}, sprintf('mode %i', modesToPlot(ii)), 'FontSize', 10, 'Rotation', 0, ...
                'VerticalAlignment', 'middle');
        case 'vertical'
            ylab = ylabel(tl3{ii}, {"mode"; modesToPlot(ii)}, 'FontSize', 10, 'Rotation', 0, ...
                'VerticalAlignment', 'middle');
    end
end

% Save figure
% savecf(sprintf("%s/results/hetero-%s_surf-%s_alpha-%.1f-%.1f_reorder-%s_visualiseModes_%i-%i", ...
%     projDir, heteroLabel, surf, alphaVals(1), alphaVals(end), reorderText, modesToPlot(1), modesToPlot(end)), ".png", 200)

%% Plot cmap
figure;
cmapCurrent = sqrt(cmaps)/1000;   % Convert units from (mm/s)^2 to m/s
cmapLimsCurrent = [min(cmapCurrent(cortexInds)), 0.85*max(cmapCurrent(cortexInds))];  % TODO: remove the 0.85
[~, ~, tl2, ~] = plotBrain('lh', {surface, medialMask, cmapCurrent}, 'colormap', viridis, 'view', {'ll', 'lm'}, ...
    'tiledLayout2Options', {1, 2, 'TileSpacing', 'none'}, 'clim', cmapLimsCurrent, 'groupBy', 'data');
cb = colorbar('southoutside'); clim(cmapLimsCurrent); colormap(gca, viridis);
cb.Layout.Tile = 'south'; 
cb.Label.String = 'm/s'; cb.Label.FontSize = 14; cb.FontSize = 14;

%% Plot correlation plot
figure; 
currentCorrs = corr(heteroModes(:, 1:nModes_vis), homoModes(:, 1:nModes_vis)); 
imagesc(abs(currentCorrs)); axis square;
cbar = colorbar(gca, "Limits", [0, 1], "Ticks", 0.1:0.1:1.0); 
cbar.FontSize = 12;
ylabel(cbar, "absolute correlation"); 
colormap(gca, turbo);
xticks(0:20:nModes); yticks(0:20:nModes);
xlabel('Homogeneous modes', 'FontSize', 14); ylabel('Heterogeneous modes', 'FontSize', 14);
% Plot box around eigengroups
for jj = 1:ceil(sqrt(size(currentCorrs, 1)))
    rectangle('Position', [(jj-1)^2+0.5, (jj-1)^2+0.5, 2*jj-1, 2*jj-1], 'EdgeColor', 'w', ...
        'LineWidth', 1.5);
end

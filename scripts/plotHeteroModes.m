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
plotCorr = 0;   % boolean for whether to plot correlation matrix
nRows = nModesToPlot + 2 + plotHist + plotCorr*3; 
nCols = nHeteroBSs + 1;

% Initialise plot
figure('Position', [100, 0, 400*nCols, 100*nRows], 'visible', 'on');
tl1 = tiledlayout(1, nHeteroBSs + 1, 'TileSpacing','compact');

% Plot geometric eigenmodes
tl2 = tiledlayout(tl1, nRows, 1, 'TileSpacing', 'tight');
tl2.Layout.Tile = 1;
[~, ~, tl3, tl4] = plotBrain('lh', {surface, medialMask, homoModes(:, modesToPlot)}, 'parent', tl2, ...
    'groupBy', 'data', 'colormap', @bluewhitered, 'tiledLayoutOptions', {nModesToPlot, 1, 'TileSpacing', 'none'}, ...
    'view', {'ll', 'lm'}, 'tiledLayout2Options', {1, 2, 'TileSpacing', 'none'});
tl3.Layout.Tile = 3 + plotHist;
tl3.Layout.TileSpan = [nModesToPlot, 1];
% Plot mode labels
for ii=1:length(modesToPlot)
    ylab = ylabel(tl4{ii}, {"mode"; modesToPlot(ii)}, 'FontSize', 10, 'Rotation', 0, ...
        'VerticalAlignment', 'middle');
end
title(tl3, 'Homogeneous modes');

% Plot eigenvalues
% nexttile(tl2, [3, 1])
% pointSize = 12;
% % Plot geometric eigvenvalues against itself
% scatter(homoEvals, homoEvals, pointSize, "filled")
% xlabel("Homogeneous eigenvalues")
% axis('square')
% legend("Homogeneous")
% hold on
% legendNames = cell(1, nHeteroBSs);
% % Plot heterogeneous eigenvalues against geometric
% for ii=1:nHeteroBSs
%     scatter(homoEvals, heteroEvals(ii, :), pointSize, "filled")
%     legendNames{ii} = sprintf('\\alpha: %.1f, \\beta: %.1f', modeParams(ii).alpha, modeParams(ii).beta);
%     hold on
% end
% legend(["Homogeneous", legendNames], "location", "northwest")

for ii = 1:nHeteroBSs
    tl2 = tiledlayout(tl1, nRows, 1, 'TileSpacing', 'compact');
    tl2.Layout.Tile = ii + 1;

    % Plot propagation speed map (C)
    cmapLimsCurrent = [min(cmaps(cortexInds, ii)), 0.85*max(cmaps(cortexInds, ii))];  % TODO: remove the 0.85
    [~, ~, tl3, ~] = plotBrain('lh', {surface, medialMask, cmaps(:, ii)}, 'parent', tl2, ...
        'groupBy', 'data', 'colormap', viridis, 'tiledLayoutOptions', {1, 1, 'TileSpacing', 'none'}, ...
        'view', {'ll', 'lm'}, 'tiledLayout2Options', {1,2, 'TileSpacing', 'none'}, ...
        'clim', cmapLimsCurrent);
    tl3.Layout.Tile = 1;
    tl3.Layout.TileSpan = [2, 1];
    title(tl3, "c_s^2 map", 'FontSize', 10)
    title(tl2, {'Heterogeneous modes'; sprintf('%s | \\alpha: %.1f, \\beta: %.1f', 'T1w/T2w',... % TOOD: make this generalisable
        modeParams(ii).alpha, modeParams(ii).beta)})
    % Plot colormap (turns out its much faster to plot the colorbar outside of the plotBrain call)
    colorbar('eastoutside'); clim(cmapLimsCurrent); colormap(gca, viridis); 
    
    % TODO: need to test that this still works
    % Plot propagation speed map distribution
    if plotHist
        ax = nexttile(tl2);
        histogram(sqrt(cmaps(cortexInds, ii)), 30)
%         title("c_s distribution")
        xlim(sqrt(cmapLimsGlobal))
        xlabel("mm/s", "FontSize", 8)
    end
    
    % Reorder heterogeneous modes according to the geometric mode that it most closely resembles
    if reorder
        [matchedModes, newAllocations, ~, ~] = matchModes(heteroModes(:, :, ii), homoModes, ...
            'withinGroups', withinGroups);
        currentModes = matchedModes;
        modeLabels = newAllocations(modesToPlot);
    else
        currentModes = heteroModes(:, :, ii);
        modeLabels = modesToPlot;
    end
    % If a hetero mode is the same as the corresponding geometric mode but just flipped then flip 
    % the sign of the hetero mode for better visual comparison
    corrs_diag = diag(corr(homoModes, currentModes)); 
    mask = corrs_diag < -0.8; 
    currentModes(:, mask) = currentModes(:, mask) * -1;
    % Plot heterogeneous modes
    [~, ~, tl3, tl4] = plotBrain('lh', {surface, medialMask, currentModes(:, modesToPlot)}, 'parent', tl2, ...
        'groupBy', 'data', 'colormap', @bluewhitered, 'tiledLayoutOptions', {nModesToPlot, 1, 'TileSpacing', 'none'},...
        'view', {'ll', 'lm'}, 'tiledLayout2Options', {1,2, 'TileSpacing', 'none'});
    tl3.Layout.Tile = 3 + plotHist;
    tl3.Layout.TileSpan = [nModesToPlot, 1];
    % If heterogeneous modes have been reordered then plot the mode labels 
    if reorder
        for jj=1:length(modeLabels)
            ylabel(tl4{jj}, sprintf("mode %i", modeLabels(jj)), 'FontSize', 10)
        end
    end

    % Plot correlationg between heterogeneous modes and geometric modes
    if plotCorr
        if reorder
            % Reorder modes
            [~, ~, ~, newCorrs] = matchModes(heteroModes(:, 1:nModes_vis, ii), homoModes(:, 1:nModes_vis), ...
                'showFigures', false, 'withinGroups', withinGroups);
            currentCorrs = newCorrs;
        else
            currentCorrs = corr(heteroModes(:, 1:nModes_vis, ii), homoModes(:, 1:nModes_vis));
        end
        % Plot matrix as heatmap
        nexttile(tl2, [3, 1])
        imagesc(abs(currentCorrs)); axis square;
        cbar = colorbar(gca, "Limits", [0, 1], "Ticks", 0.1:0.1:1.0); 
        ylabel(cbar, "absolute correlation"); 
        colormap(gca, viridis);
        xticks(0:20:nModes); yticks(0:20:nModes);
        xlabel('Homogeneous modes'); ylabel('Heterogeneous modes');
        % Plot box around eigengroups
        for jj = 1:ceil(sqrt(size(currentCorrs, 1)))
            rectangle('Position', [(jj-1)^2+0.5, (jj-1)^2+0.5, 2*jj-1, 2*jj-1]);
        end
    end
end

% Save figure
% savecf(sprintf("%s/results/hetero-%s_surf-%s_alpha-%.1f-%.1f_reorder-%s_visualiseModes_%i-%i", ...
%     projDir, heteroLabel, surf, alphaVals(1), alphaVals(end), reorderText, modesToPlot(1), modesToPlot(end)), ".png", 200)


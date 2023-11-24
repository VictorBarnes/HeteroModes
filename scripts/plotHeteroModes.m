%% Code to plot heterogeneous modes
clear
clc

% Setup project by loading necessary functions
setupProject

% Load config file
config = jsondecode(fileread("/fs04/kg98/vbarnes/HeteroModes/scripts/config.json"));
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

heteroLabel = "myelinmap"; % only plot one hetero map per figure
scale = "cmean";
alphaVals = [0.1, 0.5, 1.0];
nAlpha = length(alphaVals);

% Load Yeo surface file
[vertices, faces] = read_vtk(sprintf('%s/atlas-yeo_space-%s_den-%s_surf-%s_hemi-%s_surface.vtk', ...
    surfDir, space, den, surf, hemi));
surface.vertices = vertices';
surface.faces = faces';
% Load cortex mask
medialMask = dlmread(sprintf('%s/atlas-yeo_space-%s_den-%s_hemi-%s_medialMask.txt', surfDir, space, ...
    den, hemi));
cortexInds = find(medialMask);

geomDesc = 'hetero-%s_atlas-%s_space-%s_den-%s_surf-%s_hemi-%s_n-%i_maskMed-True';
% Load geometric eigenmodes and eigenvalues
geomModes = dlmread(fullfile(emodeDir, sprintf(geomDesc, "None", atlas, space, den, surf, hemi, nModes) ...
    + "_emodes.txt"));
geomEvals = dlmread(fullfile(emodeDir, sprintf(geomDesc, "None", atlas, space, den, surf, hemi, nModes) ...
    + "_evals.txt"));

% Load propagation speed maps (C)
heteroDesc = 'hetero-%s_atlas-%s_space-%s_den-%s_surf-%s_hemi-%s_n-%i_scale-%s_alpha-%.1f_maskMed-True';
cmaps = zeros(size(geomModes, 1), nAlpha);
for ii=1:nAlpha
    cmaps(:, ii) = dlmread(fullfile(emodeDir, "cmaps", sprintf(heteroDesc, heteroLabel, atlas, ...
        space, den, surf, hemi, nModes, scale, alphaVals(ii)) + "_cmap.txt")); 
end

% TODO: do we need this??
% Find min and max values in cmaps for plotting on using same x axis
cmapLimsGlobal = [round(min(cmaps(cortexInds, :), [], "all")), round(max(cmaps(cortexInds, :), [], "all"))];

% Load heterogeneous eigenmodes and eigenvalues
heteroModes = zeros([size(geomModes), nAlpha]);
heteroEvals = zeros(nAlpha, nModes);
for ii=1:nAlpha  
    heteroModes(:, :, ii) = dlmread(fullfile(emodeDir, sprintf(heteroDesc, heteroLabel, atlas, ...
        space, den, surf, hemi, nModes, scale, alphaVals(ii)) + "_emodes.txt")); 
    heteroEvals(ii, :) = dlmread(fullfile(emodeDir, sprintf(heteroDesc, heteroLabel, atlas, space, ...
        den, surf, hemi, nModes, scale, alphaVals(ii)) + "_evals.txt")); 
end

% Set reordering parameters
reorder = false;
withinGroups = true;
% Only save with 'withinGroups' tag in filename if `reorder` is falsealphaVals
if reorder
    reorderText = sprintf("1_withinGroups-%i", withinGroups);
else
    reorderText = num2str(reorder);
end


%% Visualize eigenmodes
% TODO: change this to plot the cmaps and distributions first then modes in one plotBrain call. This
% should make the plot look nicer and 

modesToPlot = [2, 3, 4, 10, 15, 20, 25, 50, 100, 200];
% modesToPlot = [10, 14];
nModesToPlot = length(modesToPlot);
plotHist = 1;  % boolean for whether to plot cmap distribution or not
nRows = nModesToPlot + 2 + plotHist;
nCols = nAlpha + 1;

% Initialise plot
figure('Position', [100, 0, 300*nCols, 100*nRows], 'visible', 'on');
tl1 = tiledlayout(1, nAlpha + 1, 'TileSpacing','tight');

% Plot geometric eigenmodes
tl2 = tiledlayout(tl1, nRows, 1, 'TileSpacing', 'tight');
tl2.Layout.Tile = 1;
[~, ~, tl3, tl4] = plotBrain('lh', {surface, medialMask, geomModes(:, modesToPlot)}, 'parent', tl2, ...
    'groupBy', 'data', 'colormap', @bluewhitered, 'tiledLayoutOptions', {nModesToPlot, 1, 'TileSpacing', 'none'}, ...
    'view', {'ll', 'lm'}, 'tiledLayout2Options', {1, 2, 'TileSpacing', 'none'});
tl3.Layout.Tile = 3 + plotHist;
tl3.Layout.TileSpan = [nModesToPlot, 1];
% Plot mode labels
for ii=1:length(modesToPlot)
    ylabel(tl4{ii}, sprintf("mode %i", modesToPlot(ii)), 'FontSize', 10)
end
title(tl3, 'Homogeneous modes');

for ii = 1:nAlpha
    tl2 = tiledlayout(tl1, nRows, 1, 'TileSpacing', 'tight');
    tl2.Layout.Tile = ii + 1;

    % Plot propagation speed map (C)
    cmapLimsCurrent = [min(cmaps(cortexInds, ii)), max(cmaps(cortexInds, ii))];
    [~, ~, tl3, ~] = plotBrain('lh', {surface, medialMask, cmaps(:, ii)}, 'parent', tl2, ...
        'groupBy', 'data', 'colormap', jet, 'tiledLayoutOptions', {1, 1, 'TileSpacing', 'none'}, ...
        'view', {'ll', 'lm'}, 'tiledLayout2Options', {1,2, 'TileSpacing', 'none'}, ...
        'clim', cmapLimsCurrent);
    tl3.Layout.Tile = 1;
    tl3.Layout.TileSpan = [2, 1];
    title(tl3, "c^2 map", 'FontSize', 10)
    title(tl2, {'Heterogeneous modes'; sprintf('%s | alpha: %.1f', heteroLabel, alphaVals(ii))})
    % Plot colormap
    colorbar('eastoutside'); clim(cmapLimsCurrent); colormap(gca, jet); 
    
    % Plot propagation speed map distribution
    if plotHist
        ax = nexttile(tl2);
        histogram(sqrt(cmaps(cortexInds, ii)), 30)
        title("c distribution")
        xlim(sqrt(cmapLimsGlobal))
        xlabel("mm/s", "FontSize", 8)
    end
    
    % Reorder heterogeneous modes according to the geometric mode that it most closely resembles
    if reorder
        [matchedModes, newAllocations, ~, ~] = matchModes(heteroModes(:, :, ii), geomModes, ...
            'withinGroups', withinGroups);
        currentModes = matchedModes;
        modeLabels = newAllocations(modesToPlot);
    else
        currentModes = heteroModes(:, :, ii);
        modeLabels = modesToPlot;
    end
    % If a hetero mode is the same as the corresponding geometric mode but just flipped then flip 
    % the sign of the hetero mode for better visual comparison
    corrs_diag = diag(corr(geomModes, currentModes)); 
    mask = corrs_diag < -0.8; 
    currentModes(:, mask) = currentModes(:, mask) * -1;
    % Plot heterogeneous modes
    [~, ~, tl3, ~] = plotBrain('lh', {surface, medialMask, currentModes(:, modesToPlot)}, 'parent', tl2, ...
        'groupBy', 'data', 'colormap', @bluewhitered, 'tiledLayoutOptions', {nModesToPlot, 1, 'TileSpacing', 'none'},...
        'view', {'ll', 'lm'}, 'tiledLayout2Options', {1,2, 'TileSpacing', 'none'});
    tl3.Layout.Tile = 3 + plotHist;
    tl3.Layout.TileSpan = [nModesToPlot, 1];
    % Plot mode labels for each column if they have been reordered
    if reorder
        for jj=1:length(modeLabels)
            ylabel(tl_inner{jj}, sprintf("mode %i", modeLabels(jj)), 'FontSize', 10)
        end
    end
end

% Save figure
savecf(sprintf("%s/results/hetero-%s_surf-%s_scale-%s_alpha-%.1f-%.1f_reorder-%s_visualiseModes_%i-%i", ...
    projDir, heteroLabel, surf, scale, alphaVals(1), alphaVals(end), reorderText, modesToPlot(1), modesToPlot(end)), ".png", 300)


%% Correlate heterogeneous modes with geometric modes
if nAlpha > 4
    nRows = 2;
    nCols = ceil(nAlpha/2);
    figure('Position', [200, 0, 600*nCols, 600*nRows], 'visible', 'on')
else
    nRows = 1;
    nCols = nAlpha;
    figure('Position', [200, 0, 600*nAlpha, 600], 'visible', 'on');
end

for ii=1:nAlpha
    if reorder
        % Reorder modes
        [~, ~, ~, newCorrs] = matchModes(heteroModes(:, :, ii), geomModes, 'showFigures', false, 'withinGroups', withinGroups);
        currentCorrs = newCorrs;
    else
        currentCorrs = corr(heteroModes(:, :, ii), geomModes);
    end
    
    % Plot matrix as heatmap
    subplot(nRows, nCols, ii)

    imagesc(abs(currentCorrs));
    axis square;
    cbar = colorbar; ylabel(cbar, "absolute correlation");
    xticks(0:20:200); yticks(0:20:200);
    xlabel('Homogeneous modes'); ylabel('Heterogeneous modes');
    title({'Heterogeneous eigenmodes'; sprintf("(alpha: %.1f)", alphaVals(ii))})
    % Plot box around eigengroups
    for jj = 1:ceil(sqrt(size(currentCorrs, 1))) % xline(ii^2 + 0.5);  yline(ii^2 + 0.5);
        rectangle('Position', [(jj-1)^2+0.5, (jj-1)^2+0.5, 2*jj-1, 2*jj-1]);
    end
end

% Save figure
savecf(sprintf("%s/results/hetero-%s_surf-%s_reorder-%s_scale-%s_alpha-%.1f-%.1f_corrs", ...
    projDir, heteroLabel, surf, reorderText, scale, alphaVals(1), alphaVals(end)), ".png", 300)

%% Eigenvalue plot

% Plot
figure('Position', [0, 0, 800, 800]);
pointSize = 12;

% Plot geometric eigvenvalues against itself
scatter(geomEvals, geomEvals, pointSize, "filled")
xlabel("Homogeneous eigenvalues")
title("Eigenvalue Plot")
axis('square')
legend("Homogeneous")
hold on

legendNames = cell(1, nAlpha);
% Plot heterogeneous eigenvalues against geometric
for ii=1:nAlpha
    scatter(geomEvals, heteroEvals(ii, :), pointSize, "filled")
    legendNames{ii} = sprintf("alpha: %.1f", alphaVals(ii));
    hold on
end

% set(gca, "YScale", "log")
% xlim([0, 0.05])
% ylim([0, 0.05])
legend(["Homogeneous", legendNames], "location", "northwest")

savecf(sprintf("%s/results/hetero-%s_surf-%s_scale-%s_alpha-%.1f-%.1f_eigvalPlot", ...
    projDir, heteroLabel, surf, scale, alphaVals(1), alphaVals(end)), ".png", 300)

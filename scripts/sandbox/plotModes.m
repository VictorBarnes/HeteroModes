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
surfDir = config.surface_dir;
projDir = '/fs04/kg98/vbarnes/HeteroModes';

% Load Yeo surface file
[vertices, faces] = read_vtk(fullfile(projDir, 'data', 'surfaces', sprintf('atlas-%s_space-%s_den-%s_surf-%s_hemi-%s_surface.vtk', ...
    atlas, space, den, surf, hemi)));
surface.vertices = vertices';
surface.faces = faces';
% Load cortex mask
medialMask = readmatrix(fullfile(projDir, 'data', 'surfaces', sprintf('atlas-%s_space-%s_den-%s_hemi-%s_medialMask.txt', atlas, ...
    space, den, hemi)));
cortexInds = find(medialMask);

% Load eigenmodes
emodes1 = readmatrix(fullfile(projDir, 'data', 'eigenmodes', 'hetero-None_atlas-hcp_space-fsLR_den-32k_surf-midthickness_hemi-L_n-500_alpha-0.0_beta-0.0_maskMed-True_ortho-False_emodes.txt'));
emodes2 = readmatrix(fullfile(projDir, 'data', 'eigenmodes', 'hetero-None_atlas-hcp_space-fsLR_den-32k_surf-midthickness_hemi-L_n-500_alpha-0.0_beta-0.0_maskMed-True_ortho-True_emodes.txt'));
emodes1_label = "Original eigenmodes";
emodes2_label = "B-orthonormalized eigenmdoes";


%% Plot heterogeneous modes
% TODO: change this to plot the cmaps and distributions first then modes in one plotBrain call. This
% should make the plot look nicer and 

nModes_vis = 200; % number of modes to visualise in correlation matrix
modesToPlot = [2, 3, 4, 10, 50, 100];
nModesToPlot = length(modesToPlot);
plotHist = 0;   % boolean for whether to plot cmap distribution or not
plotCorr = 1;   % boolean for whether to plot correlation matrix
plotArrangement = 'vertical';

%%% PLOT MODES (BASIS SET 1)
figure('Position', [100, 100 ,1400, 300]);
tl1 = tiledlayout('flow', 'TileSpacing','tight');
[~, ~, tl2, tl3] = plotBrain('lh', {surface, medialMask, emodes1(:, modesToPlot)}, 'parent', tl1, ...
    'groupBy', 'data', 'colormap', @bluewhitered, 'tiledLayoutOptions', {'flow', 'TileSpacing', 'tight'}, ...
    'view', {'ll', 'lm'}, 'tiledLayout2Options', {1, 2, 'TileSpacing', 'none'}, 'colorscheme', 'global');
tl2.Layout.Tile = 1;
tl2.Layout.TileSpan = [1, nModesToPlot];    % TODO
title(tl2, emodes1_label, 'FontWeight', 'bold');
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

%%% PLOT MODES (BASIS SET 2)
% If a hetero mode is the same as the corresponding geometric mode but just flipped then flip 
% the sign of the hetero mode for better visual comparison
corrs_diag = diag(corr(emodes1, emodes2)); 
mask = corrs_diag < -0.8;
emodes2(:, mask) = emodes2(:, mask) * -1;
[~, ~, tl2, tl3] = plotBrain('lh', {surface, medialMask, emodes2(:, modesToPlot)}, 'parent', tl1, ...
    'groupBy', 'data', 'colormap', @bluewhitered, 'tiledLayoutOptions', {'flow', 'TileSpacing', 'tight'}, ...
    'view', {'ll', 'lm'}, 'tiledLayout2Options', {1, 2, 'TileSpacing', 'none'}, 'colorscheme', 'global');
tl2.Layout.Tile = 2;
tl2.Layout.TileSpan = [1, nModesToPlot];    % TODO
title(tl2, emodes2_label, 'FontWeight', 'bold')     % , sprintf('%s | \\alpha: %.1f, \\beta: %.1f', 'T1w/T2w', modeParams.alpha, modeParams.beta)
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
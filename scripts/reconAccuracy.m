%% Code to understand what adding heterogeneity into eigenmodes actually does
clear
clc

% Setup project by loading necessary functions
setupProject

% Load config file
config = jsondecode(fileread(fullfile(pwd, "config.json")));
atlas = config.atlas;
space = config.space;
den = config.den;
surf = config.surf;
hemi = config.hemi;
nModes = config.n_modes;
realHeteroMaps = config.hetero_maps;
emodeDir = config.emode_dir;
surfDir = config.surface_dir;
resultsDir = config.results_dir;
pang2023dir = '/fs03/kg98/vbarnes/repos/BrainEigenmodes';

dset = "nm-subset";
heteroLabel = "myelinmap"; % only plot one hetero map per figure
scale = "cmean";
alphaVals = 0.1:0.1:1.0; 
nAlpha = length(alphaVals);

disp("Loading modes and empirical data...")
% Load Yeo surface file
[vertices, faces] = read_vtk(sprintf('%s/atlas-yeo_space-%s_den-%s_surf-%s_hemi-%s_surface.vtk', ...
    surfDir, space, den, surf, hemi));
surface.vertices = vertices';
surface.faces = faces';
% Get cortex indices
medialMask = dlmread(sprintf('%s/atlas-yeo_space-%s_den-%s_hemi-%s_medialMask.txt', surfDir, space, ...
    den, hemi));
cortexInds = find(medialMask);

% Load parcellation
parc_name = 'Glasser360';
parc = dlmread(sprintf('%s/data/parcellations/fsLR_32k_%s-lh.txt', pang2023dir, parc_name));
parcellate = false; % Whether or not to parcellate data

% Load homogeneous eigenmodes and eigenvalues
geomDesc = 'hetero-%s_atlas-%s_space-%s_den-%s_surf-%s_hemi-%s_n-%i_maskMed-True';
geomModes = dlmread(fullfile(emodeDir, sprintf(geomDesc, "None", atlas, space, den, surf, hemi, nModes) ...
    + "_emodes.txt"));
geomEvals = dlmread(fullfile(emodeDir, sprintf(geomDesc, "None", atlas, space, den, surf, hemi, nModes) ...
    + "_evals.txt"));

% Load heterogeneous eigenmodes and eigenvalues
heteroDesc = 'hetero-%s_atlas-%s_space-%s_den-%s_surf-%s_hemi-%s_n-%i_scale-%s_alpha-%.1f_maskMed-True';
heteroModes = zeros([size(geomModes), nAlpha]);
heteroEvals = zeros(nAlpha, nModes);
for ii=1:nAlpha  
    heteroModes(:, :, ii) = dlmread(fullfile(emodeDir, sprintf(heteroDesc, heteroLabel, atlas, ...
        space, den, surf, hemi, nModes, scale, alphaVals(ii)) + "_emodes.txt")); 
    heteroEvals(ii, :) = dlmread(fullfile(emodeDir, sprintf(heteroDesc, heteroLabel, atlas, space, ...
        den, surf, hemi, nModes, scale, alphaVals(ii)) + "_evals.txt")); 
end

% Load empirical data to reconstruct
if dset == "nm" || dset == "nm-subset"
    % TODO: update load_images to return images and labels separately (not as a struct)
    % Load neuromaps empiricalData
    nmData = load_images('/fs03/kg98/vbarnes/HBO_data/neuromaps-data/resampled/fsLR/dset-nm_desc-noPET_space-fsLR_den-32k_resampled.csv');
    
    if dset == "nm-subset"
        subset = {'megalpha', 'fcgradient01', 'myelinmap', 'genepc1', 'cbf'};
        nmLabels = fieldnames(nmData);
        % Find indices of subset labels within nmLabels
        indices = find(ismember(nmLabels, subset));
        % Load subset of labels and data
        empiricalLabels = nmLabels(indices);
        empiricalData = nan(size(nmData.(empiricalLabels{1}), 1), numel(empiricalLabels));
        for ii = 1:numel(empiricalLabels)
            empiricalData(:, ii) = nmData.(empiricalLabels{ii});
        end
    else
        empiricalLabels = fieldnames(nmData);
        empiricalData = nan(size(nmData.(empiricalLabels{1}), 1), numel(empiricalLabels));
        for ii = 1:numel(empiricalLabels)
            empiricalData(:, ii) = nmData.(empiricalLabels{ii});
        end
    end

    % Change labels to plotting labels for easier understanding
    configFile_HBO = fileread('/fs04/kg98/vbarnes/HumanBrainObservatory/scripts/configs/config.json');
    config_HBO = jsondecode(configFile_HBO);
    plotting_labels = config_HBO.neuromaps.plotting_labels;
    for ii = 1:length(empiricalLabels)
        empiricalLabels{ii} = plotting_labels.(empiricalLabels{ii});
    end

elseif dset == "hcp"
    % Load hcp task fMRI data
    hcpData = load(sprintf('%s/data/empirical/S255_tfMRI_ALLTASKS_raw_lh.mat', pang2023dir));
    hcpData = hcpData.zstat;

    % Only load first subject for each task contrast
    empiricalLabels = fieldnames(hcpData);
    empiricalData = nan(size(hcpData.(empiricalLabels{1}), 1), numel(empiricalLabels));
    for ii=1:numel(empiricalLabels)
        % Average across all subjects for each hcp contrast
        empiricalData(:, ii) = mean(hcpData.(empiricalLabels{jj}), 2, "omitnan");
    end
end
nImages = numel(empiricalLabels);

%% Calculate eigenreconstruction
tic
disp("Calculating eigenreconstructions...")
[geomReconCorrs, ~, ~] = calc_eigenreconstruction(empiricalData(cortexInds, :), ...
        geomModes(cortexInds, :), "handleNans");

% Calculate eigenreconstruction for hetero modes
heteroReconCorrs = nan(nModes, nImages, nAlpha);
reconCorrDiff = nan(nModes, nImages, nAlpha);
for ii = 1:nAlpha
    fprintf("alpha: %.1f", alphaVals(ii))
    [heteroReconCorrs(:, :, ii), ~, ~] = calc_eigenreconstruction(empiricalData(cortexInds, :), ...
        heteroModes(cortexInds, :, ii), "handleNans");
    reconCorrDiff(:, :, ii) = heteroReconCorrs(:, :, ii) - geomReconCorrs;
end
fprintf("Finished computing reconstruction difference: %.2f mins\n", toc/60)


%% Plot recon accuracy using stacked bar chart
modeReconPoints = [5, 10, 20, 50, 100];
plotTargetMaps = 1;

for ii = 1:nAlpha
    reconCombinedStacked = nan(nImages, 2, length(modeReconPoints));
    
    for jj = 1:length(modeReconPoints)
        if jj == 1
            prevAccGeom = zeros([1, nImages]);
            prevAccHetero = zeros([1, nImages]);
        else
            prevAccGeom = geomReconCorrs(modeReconPoints(jj - 1), :);
            prevAccHetero = heteroReconCorrs(modeReconPoints(jj - 1), :, ii);
        end
        % Subtract corr accuracy at previous modeReconPoint from the corr accuracy at the current
        % modeReconPoint
        reconCombinedStacked(:, 1, jj) = geomReconCorrs(modeReconPoints(jj), :) - prevAccGeom;
        reconCombinedStacked(:, 2, jj) = heteroReconCorrs(modeReconPoints(jj), :, ii) - prevAccHetero; 
    end

    figure('Position', [200, 200, 1600, 800], 'visible', 'off');
    nRows = 6;
    tl1 = tiledlayout(nRows, 1, 'TileSpacing', 'none');
    title(tl1, sprintf("hetero: %s | alpha: %.1f", heteroLabel, alphaVals(ii)))

    % Plot the target maps
    tl2 = tiledlayout(tl1, 1, nImages, 'TileSpacing', 'tight');
    tl2.Layout.Tile = 1;
    for jj = 1:nImages
        % Calculate clims based on min and max values of cortexInds. Some maps have a bunch of zero
        % values within the cortex indices. Omit these from the clims calculations.
        cortexData = empiricalData(cortexInds, jj);
        isZero = (abs(cortexData) < 1e-4);
        nonZeroData = cortexData(~isZero);
        clims = [min(nonZeroData), max(nonZeroData)];
        % Hard code max colour limit for specific maps for better visualisation
        if empiricalLabels{jj} == "T1w/T2w"
            clims(2) = 1.7;
        elseif empiricalLabels{jj} == "CBF"
            clims(1) = 4000;
        end

        [~, ~, tl3, tl4] = plotBrain('lh', {surface, medialMask, empiricalData(:, jj)}, 'parent', tl2, ...
            'colormap', viridis, 'view', {'ll', 'lm'}, 'tiledLayoutOptions', {1, 1, 'TileSpacing', 'none'}, ...
            'tiledLayout2Options', {1, 2, 'TileSpacing', 'none'}, 'clim', clims, 'groupBy', 'data'); 
        tl3.Layout.Tile = jj;
    end

    % Plot the recon accuracy plot
    ax = nexttile(tl1, [nRows-1, 1]);
    h = plotBarStackGroups(reconCombinedStacked, empiricalLabels);
    
    % TODO: use make_cmap('red', 5)
    blue_map = cat(1, [239, 243, 255]/255, [189,215,231]/255, [107,174,214]/255, [49,130,189]/255, [8,81,156]/255);
    red_map = cat(1, [254,229,217]/255, [252,174,145]/255, [251,106,74]/255, [222,45,38]/255, [165,15,21]/255);
    
    set(h(:,:), 'FaceColor', 'Flat')
    for jj = 1:size(blue_map, 1)
        h(1, jj).CData = blue_map(jj, :);
        h(2, jj).CData = red_map(jj, :);
    end
    
    xLabs = get(ax, 'XTickLabel');
    set(ax, 'XTickLabel', xLabs, 'FontSize', 12);
    xlabel("Reconstructed map", "FontSize", 15)
    ylabel("Reconstruction accuracy", "FontSize", 15) 
    lgd = legend('5 modes', '10 modes', '20 modes', '50 modes', '100 modes', ...
        '5 modes', '10 modes', '20 modes', '50 modes', '100 modes');
    lgd.NumColumns = 2; lgd.Location = "eastoutside"; lgd.FontSize = 12;
    title(lgd, "Homogeneous    Heterogeneous", 'FontSize', 15)
    
    % Save figure
    savecf(sprintf("%s/recon/hetero-%s_dset-%s_surf-%s_scale-%s_alpha-%.1f_reconAccuracy", ...
        resultsDir, heteroLabel, dset, surf, scale, alphaVals(ii)), ".png", 150)
end

%% Plot recon corr paths (for geom and hetero) and difference between recon corrs

% Set the colormap limits for all plots
clims = [min(reconCorrDiff(2:nModes, :, :), [], "all"), max(reconCorrDiff(2:nModes, :, :), [], "all")];

for ii = 1:nAlpha
    fig = figure('Position', [200, 200, 1500, 900], 'visible', 'off');
    tl1 = tiledlayout(2, 2);
    title(tl1, {"Reconstruction accuracy"; sprintf('(hetero: %s | alpha: %.1f)', heteroLabel, alphaVals(ii))}, 'interpreter', 'none')

    % Plot recon corr paths for hetero modes
    nexttile
    hold on; 
    for ii=1:nImages
        plot(heteroReconCorrs(2:nModes, ii, ii)); 
    end
    hold off; 
    xlabel("number of modes"); title("Reconstruction accuracy (heterogeneous)", 'fontweight', 'normal');
    ylabel("reconstruction accuracy");
    ylim([0, 1])
%     legend(labels, "location", "southeast", 'Interpreter', 'none')

    % Plot reconstruction accuracy difference
    ax = nexttile([2, 1]);
    imagesc(reconCorrDiff(2:nModes, :, ii).')  % transpose matrix to have nModes along the x-axis
    % Plot horizontal lines of separation for better visualisation
    for ii=1:nImages-1
        yline(ii+0.5, 'k-');
    end
    % Set tile parameters
    clim(clims)
    colormap(bluewhitered_mg)
    set(gca, 'ticklength', [0.02, 0.02], 'xlim', [1, nModes], 'ytick', 1:nImages,...
        'yticklabel', empiricalLabels, 'ticklabelinterpreter', 'none')
    xlabel('number of modes')
    title({'Reconstruction accuracy difference'; '(heterogeneous - homogeneous)'}, 'fontweight', 'normal')
    cbar = colorbar('Location', 'eastoutside');
    ylabel(cbar, 'reconstruction accuracy difference')

    % Plot recon corr paths for hetero modes
    nexttile
    hold on; 
    for ii=1:nImages
        plot(geomReconCorrs(2:nModes, ii)); 
    end
    hold off; 
    xlabel("number of modes"); title("Reconstruction accuracy (homogeneous)", 'fontweight', 'normal');
    ylabel("reconstruction accuracy");
    ylim([0, 1])
%     legend(labels, "location", "southeast", 'Interpreter', 'none')
    
    % Save figure
    savecf(sprintf("%s/evaluation/hetero-%s_dset-%s_surf-%s_scale-%s_alpha-%.1f_reconAccuracy", ...
        resultsDir, heteroLabel, dset, surf, scale, alphaVals(ii)), ".png", 200)
end

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

heteroLabel = "myelinmap"; % only plot one hetero map per figure
scale = "cmean";
alphaVals = 0.1:0.1:1.0;
dset = "hcp";

% TODO: set list of basis set filenames
nBasisSets = length(alphaVals);
basisSetLabels = alphaVals;

disp("Loading modes and empirical data...")

% Get cortex indices
medialMask = dlmread(sprintf('%s/atlas-yeo_space-%s_den-%s_hemi-%s_medialMask.txt', surfDir, space, ...
    den, hemi));
cortexInds = find(medialMask);

% Load parcellation
parc_name = 'Glasser360';
parc = dlmread(sprintf('%s/data/parcellations/fsLR_32k_%s-lh.txt', pang2023dir, parc_name));
parcellate = true; % Whether or not to parcellate data

% Load geometric eigenmodes and eigenvalues
geomDesc = 'hetero-%s_atlas-%s_space-%s_den-%s_surf-%s_hemi-%s_n-%i_maskMed-True';
geomModes = dlmread(fullfile(emodeDir, sprintf(geomDesc, "None", atlas, space, den, surf, hemi, nModes) ...
    + "_emodes.txt"));
geomEvals = dlmread(fullfile(emodeDir, sprintf(geomDesc, "None", atlas, space, den, surf, hemi, nModes) ...
    + "_evals.txt"));

% Load heterogeneous eigenmodes and eigenvalues
heteroDesc = 'hetero-%s_atlas-%s_space-%s_den-%s_surf-%s_hemi-%s_n-%i_scale-%s_alpha-%.1f_maskMed-True';
heteroModes = zeros([size(geomModes), nBasisSets]);
heteroEvals = zeros(nBasisSets, nModes);
for ii=1:nBasisSets  
    heteroModes(:, :, ii) = dlmread(fullfile(emodeDir, sprintf(heteroDesc, heteroLabel, atlas, ...
        space, den, surf, hemi, nModes, scale, alphaVals(ii)) + "_emodes.txt")); 
    heteroEvals(ii, :) = dlmread(fullfile(emodeDir, sprintf(heteroDesc, heteroLabel, atlas, space, ...
        den, surf, hemi, nModes, scale, alphaVals(ii)) + "_evals.txt")); 
end

% Load empirical data to reconstruct
if dset == "nm" || dset == "nm-subset"
    % TODO: update load_images to return images and labels separately (not as a struct)
    % Load neuromaps empiricalData
    nm_data = load_images('/fs03/kg98/vbarnes/HBO_data/neuromaps-data/resampled/fsLR/dset-nm_desc-noPET_space-fsLR_den-32k_resampled.csv');
    
    % TODO: modify `calc_eigendecomposition` to allow for nans but remove `abagen` until then
    empiricalLabels = fieldnames(nm_data);
    empiricalData = nan(size(nm_data.(empiricalLabels{1}), 1), numel(empiricalLabels));
    for ii = 1:numel(empiricalLabels)
        empiricalData(:, ii) = hcpData.(empiricalLabels{ii});
    end

    % TODO: fix this
    if dset == "nm-subset"
        all_empiricalData = empiricalData;
        % Get subset of neuromaps data
        subset = {'megalpha', 'fcgradient01', 'myelinmap', 'evoexp_xu2020',...
            'scalinghcp'};
        empiricalData = struct();
        for ii = 1:numel(subset)
            label = subset{ii};
            empiricalData.(label) = all_empiricalData.(label);
        end
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
        empiricalData(:, ii) = mean(hcpData.(empiricalLabels{ii}), 2, "omitnan");
    end
end
nImages = numel(empiricalLabels);

%% Calculate eigenreconstruction for geometric modes
disp("Calculating eigenreconstructions...")

[geomReconCorrs, ~, ~] = calc_eigenreconstruction(empiricalData(cortexInds, :), ...
        geomModes(cortexInds, :), "matrix");

% Calculate eigenreconstruction for hetero modes
heteroReconCorrs = nan(nModes, nImages, length(alphaVals));
reconCorrDiff = nan(nModes, nImages, length(alphaVals));
for ii = 1:length(alphaVals)
    fprintf("alpha: %.1f", alphaVals(ii))
    tic
    [heteroReconCorrs(:, :, ii), ~, ~] = calc_eigenreconstruction(empiricalData(cortexInds, :), ...
        heteroModes(cortexInds, :, ii), "matrix");
    reconCorrDiff(:, :, ii) = heteroReconCorrs(:, :, ii) - geomReconCorrs;

    fprintf("Finished computing reconstruction difference: %.2f mins\n", toc/60)

    % TODO: remove this since we now plot in another for loop so we can take the overall clims of
    % reconCorrDiff
    % Plot difference between reconstruction correlation results
    % fig = figure('Position', [200, 200, 1500, 900]);
    % % Set colormap limits for plotting
    % clim_max = max(reconCorrDiff(:, 2:end, ii), [], 'all');
    % clim_min = min(reconCorrDiff(:, 2:end, ii), [], 'all');
    % clims = [clim_min, clim_max];
    % 
    % % Plot reconstruction accuracy difference
    % imagesc(reconCorrDiff(:, 2:end, ii).')  % transpose matrix to have nModes along the x-axis
    % % Plot horizontal lines of separation for better visualisation
    % for jj=1:nImages-1
    %     yline(jj+0.5, 'k-');
    % end
    % 
    % clim(clims)
    % colormap(bluewhitered)
    % % cbar = colorbar;
    % set(gca, 'fontsize', 14, 'ticklength', [0.02, 0.02], 'xlim', [1, nModes], 'ytick', 1:nImages,...
    %     'yticklabel', empiricalLabels, 'ticklabelinterpreter', 'none')
    % xlabel('number of modes')
    % title({'Reconstruction accuracy difference (heterogeneous - geometric)'; sprintf('(hetero: %s | alpha: %.1f)', heteroLabel, alphaVals(ii))}, 'fontweight', 'normal', 'fontsize', 18)
    % 
    % cbar = colorbar('Location', 'eastoutside');
    % ylabel(cbar, 'reconstruction accuracy difference', 'fontsize', 16)
    % 
    % % Save figure
    % savecf(sprintf("%s/evaluation/hetero-%s_surf-%s_scale-%s_alpha-%.1f_empiricalData-%s_reconAccuracyDiff", ...
    %     resultsDir, heteroLabel, surf, scale, alphaVals(ii), dset), ".png", 200)
end

%% Plot recon corr paths (for geom and hetero) and difference between recon corrs

% Set the colormap limits for all plots
clims = [min(reconCorrDiff(:)), max(reconCorrDiff(:))];

for ii = 1:length(alphaVals)
    fig = figure('Position', [200, 200, 1500, 900]);
    tl = tiledlayout(2, 2);
    title(tl, {"Reconstruction accuracy"; sprintf('(hetero: %s | alpha: %.1f)', heteroLabel, alphaVals(ii))})

    % Plot recon corr paths for hetero modes
    nexttile
    hold on; 
    for jj=1:nImages
        plot(heteroReconCorrs(2:nModes, jj, ii)); 
    end
    hold off; 
    xlabel("number of modes"); title("Reconstruction accuracy (heterogeneous)", 'fontweight', 'normal');
    ylabel("reconstruction accuracy");
    ylim([0, 1])
%     legend(labels, "location", "southeast", 'Interpreter', 'none')

    % Plot reconstruction accuracy difference
    ax = nexttile([2, 1]);
    imagesc(reconCorrDiff(2:end, :, ii).')  % transpose matrix to have nModes along the x-axis
    % Plot horizontal lines of separation for better visualisation
    for jj=1:nImages-1
        yline(jj+0.5, 'k-');
    end  
    % Set tile parameters
    clim(clims)
    colormap(bluewhitered)
    set(gca, 'ticklength', [0.02, 0.02], 'xlim', [1, nModes], 'ytick', 1:nImages,...
        'yticklabel', empiricalLabels, 'ticklabelinterpreter', 'none')
    xlabel('number of modes')
    title({'Reconstruction accuracy difference'; '(heterogeneous - geometric)'}, 'fontweight', 'normal')
    cbar = colorbar('Location', 'eastoutside');
    ylabel(cbar, 'reconstruction accuracy difference')

    % Plot recon corr paths for hetero modes
    nexttile
    hold on; 
    for jj=1:nImages
        plot(geomReconCorrs(2:nModes, jj)); 
    end
    hold off; 
    xlabel("number of modes"); title("Reconstruction accuracy (geometric)", 'fontweight', 'normal');
    ylabel("reconstruction accuracy");
    ylim([0, 1])
%     legend(labels, "location", "southeast", 'Interpreter', 'none')
    
    % Save figure
    savecf(sprintf("%s/evaluation/hetero-%s_surf-%s_scale-%s_alpha-%.1f_empiricalData-%s_reconAccuracy", ...
        resultsDir, heteroLabel, surf, scale, alphaVals(ii), dset), ".png", 200)
end

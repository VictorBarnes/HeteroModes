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
BEdir = '/fs03/kg98/vbarnes/repos/BrainEigenmodes';

dset = "nm-subset";
heteroLabel = "myelinmap"; % only plot one hetero map per figure
scale = "cmean";
alphaVals = 1.0;
beta = -0.5;
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
parc = dlmread(sprintf('%s/data/parcellations/fsLR_32k_%s-lh.txt', BEdir, parc_name));
nParcels = length(unique(parc(parc>0)));

% Load homogeneous eigenmodes and eigenvalues
geomDesc = 'hetero-%s_atlas-%s_space-%s_den-%s_surf-%s_hemi-%s_n-%i_maskMed-True';
geomModes = dlmread(fullfile(emodeDir, sprintf(geomDesc, "None", atlas, space, den, surf, hemi, nModes) ...
    + "_emodes.txt"));
geomEvals = dlmread(fullfile(emodeDir, sprintf(geomDesc, "None", atlas, space, den, surf, hemi, nModes) ...
    + "_evals.txt"));

% Load heterogeneous eigenmodes and eigenvalues
heteroDesc = 'hetero-%s_atlas-%s_space-%s_den-%s_surf-%s_hemi-%s_n-%i_scale-%s_alpha-%.1f_beta-%.1f_maskMed-True';
heteroModes = zeros([size(geomModes), nAlpha]);
heteroEvals = zeros(nAlpha, nModes);
for ii=1:nAlpha  
    heteroModes(:, :, ii) = dlmread(fullfile(emodeDir, sprintf(heteroDesc, heteroLabel, atlas, ...
        space, den, surf, hemi, nModes, scale, alphaVals(ii), beta) + "_emodes.txt")); 
    heteroEvals(ii, :) = dlmread(fullfile(emodeDir, sprintf(heteroDesc, heteroLabel, atlas, space, ...
        den, surf, hemi, nModes, scale, alphaVals(ii), beta) + "_evals.txt")); 
end

% Load example single-subject rfMRI time series data
rsfMRI = load(fullfile(BEdir, 'data', 'examples', 'subject_rfMRI_timeseries-lh.mat'));
empTimeSeries = rsfMRI.timeseries;
T = size(empTimeSeries, 2);
% Extract upper triangle indices
triuInds = calc_triu_ind(zeros(nParcels, nParcels));

%% Calculate FC reconstruction accuracy
disp("Calculating FC reconstruction accuracy")
% Calculate empirical FC
empTimeSeries_parc = calc_normalize_timeseries(calc_parcellate(parc, empTimeSeries));
empTimeSeries_parc(isnan(empTimeSeries_parc)) = 0;
empFC = (empTimeSeries_parc'*empTimeSeries_parc)/T;

% Calculate eigenreconstructions
[~, geomRecon, ~] = calc_eigenreconstruction(empTimeSeries(cortexInds, :), ...
    geomModes(cortexInds, :), "handleNans", false);
% [~, heteroRecon, ~] = calc_eigenreconstruction(empTimeSeries(cortexInds, :), ...
%     heteroModes(cortexInds, :, ii), "handleNans");

% Calculate reconstructed FC and accuracy (slow to run with more modes)
geomReconFC_parc = zeros(length(triuInds), nModes);
heteroReconFC_parc = zeros(length(triuInds), nModes);
geomReconCorr = zeros(1, nModes);
heteroReconCorr = zeros(1, nModes);
for mode=1:nModes
    geomRecon_parc = calc_normalize_timeseries(calc_parcellate(parc(cortexInds), squeeze(geomRecon(:, mode, :))));
    geomRecon_parc(isnan(geomRecon_parc)) = 0;
    geomReconFC = (geomRecon_parc'*geomRecon_parc)/T;
    
    geomReconFC_parc(:,mode) = geomReconFC(triuInds);         
    geomReconCorr(mode) = corr(empFC(triuInds), geomReconFC_parc(:,mode));
end

%% Plot results
% Reconstruction accuracy vs number of modes at parcellated level
figure('Name', 'rfMRI reconstruction - accuracy');
hold on;
plot(1:nModes, geomReconCorr)
% plot(1:nModes, heteroReconCorr)
hold off;
set(gca, 'fontsize', 10, 'ticklength', [0.02 0.02], 'xlim', [1 nModes], 'ylim', [0 1])
xlabel('number of modes', 'fontsize', 12)
ylabel('reconstruction accuracy', 'fontsize', 12)

% Reconstructed FC using N = num_modes modes
FC_recon = zeros(nParcels, nParcels);
FC_recon(triuInds) = geomReconFC_parc(:, nModes);
FC_recon = FC_recon + FC_recon';
FC_recon(1:(nParcels+1):nParcels^2) = 1;

figure('Name', sprintf('rfMRI reconstruction - FC matrix using %i modes', nModes));
imagesc(FC_recon)
caxis([-1 1])
colormap(bluewhitered)
cbar = colorbar;
set(gca, 'fontsize', 10, 'ticklength', [0.02 0.02])
xlabel('region', 'fontsize', 12)
ylabel('region', 'fontsize', 12)
ylabel(cbar, 'FC', 'fontsize', 12)
axis image
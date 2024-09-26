%% Simulate wave model
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
BEdir = '/fs03/kg98/vbarnes/repos/BrainEigenmodes';

% Load Yeo surface file
[vertices, faces] = read_vtk(sprintf('%s/atlas-S1200_space-%s_den-%s_surf-%s_hemi-%s_surface.vtk', ...
    surfDir, space, den, surf, hemi));
surface.vertices = vertices';
surface.faces = faces';
% Load cortex mask
medialMask = dlmread(sprintf('%s/atlas-S1200_space-%s_den-%s_hemi-%s_medialMask.txt', surfDir, space, ...
    den, hemi));
cortexInds = find(medialMask);
medialInds = find(~medialMask);
nVerts_full = size(surface.vertices, 1);
nVerts_mask = length(cortexInds);

emodes1 = readmatrix('/fs04/kg98/vbarnes/HeteroModes/data/eigenmodes/hetero-None_atlas-S1200_space-fsLR_den-32k_surf-midthickness_hemi-L_n-500_maskMed-True_emodes.txt');
evals1 = readmatrix('/fs04/kg98/vbarnes/HeteroModes/data/eigenmodes/hetero-None_atlas-S1200_space-fsLR_den-32k_surf-midthickness_hemi-L_n-500_maskMed-True_evals.txt');
label1 = 'homogeneous';

emodes2 = readmatrix('/home/vbarnes/kg98/vbarnes/HeteroModes/data/eigenmodes/hetero-centralPatch/hetero-centralPatch_atlas-S1200_space-fsLR_den-32k_surf-midthickness_hemi-L_alpha-1.0_beta-1.0_n-500_maskMed-True_ortho-False_n-500_emodes.txt');
evals2 = readmatrix('/home/vbarnes/kg98/vbarnes/HeteroModes/data/eigenmodes/hetero-centralPatch/hetero-centralPatch_atlas-S1200_space-fsLR_den-32k_surf-midthickness_hemi-L_alpha-1.0_beta-1.0_n-500_maskMed-True_ortho-False_n-500_evals.txt');
label2 = 'central patch';
% emodes1 = readmatrix('/fs04/kg98/vbarnes/HeteroModes/data/eigenmodes/hetero-None_atlas-S1200_space-fsLR_den-32k_surf-midthickness_hemi-L_alpha-0.0_beta-0.0_n-500_maskMed-True_ortho-False_n-500_maskMed-True_ortho-False_emodes.txt');
% evals1 = readmatrix('/fs04/kg98/vbarnes/HeteroModes/data/eigenmodes/hetero-None_atlas-S1200_space-fsLR_den-32k_surf-midthickness_hemi-L_alpha-0.0_beta-0.0_n-500_maskMed-True_ortho-False_n-500_maskMed-True_ortho-False_evals.txt');
% homoModes = readmatrix('/home/vbarnes/kg98_scratch/jamesp/eigenmode_templates/fsLR_32k_midthickness-lh_evec_500.txt');
% homoEvals = readmatrix('/home/vbarnes/kg98_scratch/jamesp/eigenmode_templates/fsLR_32k_midthickness-lh_eval_500.txt');

% heteroLabel = 'centralPatch';
% emodes2 = readmatrix('/fs04/kg98/vbarnes/HeteroModes/data/eigenmodes/hetero-centralPatch/hetero-centralPatch_atlas-S1200_space-fsLR_den-32k_surf-midthickness_hemi-L_alpha-1.0_beta-1.0_n-500_maskMed-True_ortho-False_n-500_maskMed-True_ortho-False_emodes.txt');
% evals2 = readmatrix('/fs04/kg98/vbarnes/HeteroModes/data/eigenmodes/hetero-centralPatch/hetero-centralPatch_atlas-S1200_space-fsLR_den-32k_surf-midthickness_hemi-L_alpha-1.0_beta-1.0_n-500_maskMed-True_ortho-False_n-500_maskMed-True_ortho-False_evals.txt');
% cmaps = readmatrix(fullfile(emodeDir, "cmaps", sprintf(heteroDesc, heteroLabel, atlas, space, den, surf, hemi, nModes, scale, alphaVals(ii)) + "_cmap.txt")); 

%% Simulate evoked activity (stimulating V1)

% =========================================================================
%                    Load the HCPMMP1 (Glasser360) atlas                   
% =========================================================================

parc = gifti(fullfile(BEdir, 'data', 'parcellations', ...
    sprintf('Q1-Q6_RelatedParcellation210.%s.CorticalAreas_dil_Colors.32k_fs_LR.label.gii', hemi)));

% =========================================================================
%               Find parcel number and index of vertices of V1             
% =========================================================================

ROI = sprintf("%s_V1_ROI", hemi);
parcel_number = parc.labels.key(strcmpi(parc.labels.name, ROI));
% parcel_ind = find(parc.cdata==parcel_number);
parcel_ind = find(parc.cdata(cortexInds)==parcel_number);

% =========================================================================
%      Simulate evoked activity stimulating all vertices within V1 ROI     
% =========================================================================

param = loadParameters_wave_func;
param.tstep = 0.1; % in ms
param.tmax = 100;  % in ms
param.tspan = [0, param.tmax];
param.T = 0:param.tstep:param.tmax;

% Change value of param.is_time_ms to 1 because the time defined above is
% in ms. This is necessary as param.gamma_s needs to match the scale.
param.is_time_ms = 1;

% Method for solving the wave model (either 'ODE' or 'Fourier')
% 'Fourier' is faster for long time series
% method = 'ODE';
method = 'ODE';

param.r_s = 28.9;       % (default) in mm
param.gamma_s = 116;    % (default) in s^-1
if param.is_time_ms==1
    param.gamma_s = 116*1e-3;
end

% Load mass matrix
param.B = load(fullfile(projDir, 'data', 'Bmatrix.mat')).B(cortexInds, cortexInds);

% Create a 1 ms external input with amplitude = 20 to V1 ROI
% results are robust to amplitude
ext_input_time_range = [1, 2];
ext_input_time_range_ind = dsearchn(param.T', ext_input_time_range');
ext_input_amplitude = 20;

ext_input = zeros(nVerts_mask, length(param.T));
ext_input(parcel_ind, ext_input_time_range_ind(1):ext_input_time_range_ind(2)) = ext_input_amplitude; 

param.hetero = 0;
[~, homoSimActivity] = model_neural_waves(emodes1(cortexInds, :), evals1, ext_input, param, method);
param.hetero = 1;
[~, heteroSimActivity] = model_neural_waves(emodes2(cortexInds, :), evals2, ext_input, param, method);

% Reshape sim activity to include medial wall
homoSimActivityFull = nan(nVerts_full, length(param.T));
homoSimActivityFull(cortexInds, :) = homoSimActivity;
heteroSimActivityFull = nan(nVerts_full, length(param.T));
heteroSimActivityFull(cortexInds, :) = heteroSimActivity;

%% Plot video of simulated activity

% Video of activity every 0.5 ms (increase this to better see the waves) t = param.T;
tInterest = 0:0.5:param.tmax;
is_time_ms = 1;
with_medial = 0;
cmap = parula;
show_colorbar = 1;
output_filename = sprintf("%s/results/sim/simActivityVid", projDir);
save_video = 0;

fig = video_surface_activity(surface, heteroSimActivityFull, 'lh', param.T, ...
                             tInterest, is_time_ms, medialInds, with_medial, ...
                             cmap, show_colorbar, output_filename, save_video);

%% Visualize results                      

% Snapshot of activity snapshot every 10 ms
tInterest = linspace(10, param.tmax, 10);
tInterestInds = dsearchn(param.T', tInterest');

% Initialise figure
nRows = 2;
nCols = length(tInterest);
figure('Position', [200, 200, 200*nCols, 200*nRows], 'visible', 'on');
tl = tiledlayout(nRows, 1, 'TileSpacing', 'loose');

% Plot simulated activity calculated on the geometric modes
tlRow = tiledlayout(tl, 1, nCols);
tlRow.Layout.Tile = 1;
[~, ~, tlhomo, ~] = plotBrain('lh', {surface, medialMask, real(log10(homoSimActivityFull(:, tInterestInds)))}, ...
    'parent', tlRow, 'tiledLayoutOptions', {1, length(tInterest), 'TileSpacing', 'none'}, ...
    'tiledLayout2Options', {2, 1, 'TileSpacing', 'none'}, 'view', {'ll', 'lm'}, 'groupBy', 'data', 'colorscheme', 'global');
tlhomo.Layout.TileSpan = [1, length(tInterest)];
title(tlhomo, label1)

% Plot heterogeneity map with simulated activity calculated on the heterogeneous modes
tlRow = tiledlayout(tl, 1, nCols);
tlRow.Layout.Tile = 2;
[~, ~, tl2, tl3] = plotBrain('lh', {surface, medialMask, real(log10(heteroSimActivityFull(:, tInterestInds)))}, 'parent', tlRow, ...
    'view', {'ll', 'lm'}, 'tiledLayoutOptions', {1, nCols, 'TileSpacing', 'none'}, ...
    'tiledLayout2Options', {2, 1, 'TileSpacing', 'none'}, 'groupBy', 'data', 'colorscheme', 'global');
tl2.Layout.TileSpan = [1, nCols];
title(tl2, label2)
% Set colormap and label for c2 map
% c2_axs = tl3{1}.Children;
% colormap(c2_axs(1), viridis); colormap(c2_axs(2), viridis)
% xlabel(tl3{1}, 'c_s^2 map')
% % Plot time labels
% for jj = 1:length(tInterest)
%     xlabel(tl3{jj+1}, sprintf('%.1f ms', tInterest(jj)))
% end

% Save figure
% savecf(sprintf("%s/results/sim/hetero-%s_surf-%s_scale-%s_alpha-%.1f-%.1f_simActivity", ...
%     projDir, heteroLabel, surf, scale, alphaVals(1), alphaVals(end)), ".png", 200)


%% View results from Pang2023
data_results_folder = fullfile(BEdir, "data", "results");
model_wave_visual = load(sprintf('%s/model_wave_neural_visual_%s.mat', data_results_folder, hemisphere), ...
                                 'tspan', 'simulated_neural_visual_vertex', 'simulated_neural_visual_parcel', ...
                                 'ROIS', 'ROI_names');
figure; imagesc(model_wave_visual.simulated_neural_visual_vertex)


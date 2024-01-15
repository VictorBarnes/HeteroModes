%% Simulate wave model
% TODO: change geometric to homogeneous
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

heteroLabel = "myelinmap"; % only plot one hetero map per figure
scale = "cmean";
alphaVals = [1.0];
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
medialInds = find(~medialMask);

geomDesc = 'hetero-%s_atlas-%s_space-%s_den-%s_surf-%s_hemi-%s_n-%i_scale-%s_maskMed-True';
% Load geometric eigenmodes and eigenvalues
geomModes = dlmread(fullfile(emodeDir, sprintf(geomDesc, "None", atlas, space, den, surf, hemi, nModes, scale) ...
    + "_emodes.txt"));
geomEvals = dlmread(fullfile(emodeDir, sprintf(geomDesc, "None", atlas, space, den, surf, hemi, nModes, scale) ...
    + "_evals.txt"));

% Load modes derived by Pang2023 for comparison
pangModes = dlmread("/home/vbarnes/kg98_scratch/vbarnes/repos/BrainEigenmodes/data/template_eigenmodes/fsLR_32k_midthickness-lh_emode_200.txt");
pangEvals = dlmread("/home/vbarnes/kg98_scratch/vbarnes/repos/BrainEigenmodes/data/template_eigenmodes/fsLR_32k_midthickness-lh_eval_200.txt");
pangModes500 = dlmread("/home/vbarnes/kg98_scratch/jamesp/eigenmode_templates/fsLR_32k_midthickness-lh_evec_500.txt");
pangEvals500 = dlmread("/home/vbarnes/kg98_scratch/jamesp/eigenmode_templates/fsLR_32k_midthickness-lh_eval_500.txt");

heteroDesc = 'hetero-%s_atlas-%s_space-%s_den-%s_surf-%s_hemi-%s_n-%i_scale-%s_alpha-%.1f_maskMed-True';
% Load heterogeneity map
cmaps = zeros(size(geomModes, 1), nAlpha);
for ii=1:nAlpha
    cmaps(:, ii) = dlmread(fullfile(emodeDir, "cmaps", sprintf(heteroDesc, heteroLabel, atlas, ...
        space, den, surf, hemi, nModes, scale, alphaVals(ii)) + "_cmap.txt")); 
end
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
parcel_ind = find(parc.cdata==parcel_number);

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
method = 'Fourier';

param.r_s = 28.9;      % 30 (default) in mm
param.gamma_s = 116; % (default) in s^-1
if param.is_time_ms==1
    param.gamma_s = 116*1e-3;
end

% Create a 1 ms external input with amplitude = 20 to V1 ROI
% results are robust to amplitude
ext_input_time_range = [1, 2];
ext_input_time_range_ind = dsearchn(param.T', ext_input_time_range');
ext_input_amplitude = 20;

ext_input = zeros(size(geomModes,1), length(param.T));
ext_input(parcel_ind, ext_input_time_range_ind(1):ext_input_time_range_ind(2)) = ext_input_amplitude; 

[~, geomSimActivity] = model_neural_waves(geomModes, geomEvals, ext_input, param, method);
a = 1;
heteroSimActivity = zeros(size(geomModes, 1), length(param.T), nAlpha);
for ii = 1:nAlpha
    [~, heteroSimActivity(:, :, ii)] = model_neural_waves(heteroModes(:, :, a), heteroEvals(a, :), ext_input, param, method);
end

%% Visualize results                      

% Snapshot of activity snapshot every 10 ms
tInterest = linspace(10, param.tmax, 10);
tInterestInds = dsearchn(param.T', tInterest');

% Initialise figure
nRows = length(alphaVals) + 1;
nCols = length(tInterest) + 1;
figure('Position', [200, 200, 200*nCols, 200*nRows], 'visible', 'on');
tl = tiledlayout(nRows, 1, 'TileSpacing', 'loose');

% Plot simulated activity calculated on the geometric modes
tlRow = tiledlayout(tl, 1, nCols);
[~, ~, tlgeom, ~] = plotBrain('lh', {surface, medialMask, geomSimActivity(:, tInterestInds)}, ...
    'groupBy', 'data', 'parent', tlRow, 'tiledLayoutOptions', {1, length(tInterest), 'TileSpacing', 'none'}, ...
    'tiledLayout2Options', {2, 1, 'TileSpacing', 'none'}, 'view', {'ll', 'lm'});
tlgeom.Layout.Tile = 2;
tlgeom.Layout.TileSpan = [1, length(tInterest)];
title(tlgeom, 'Simulated activity on homogeneous modes')

for ii = 1:length(alphaVals)
    % Combine heterogeneity map and simulated activity into one 2d array for plotting
    mapsToPlot = cat(2, cmaps(:, ii), heteroSimActivity(:, tInterestInds, ii));
    % Plot heterogeneity map with simulated activity calculated on the heterogeneous modes
    tlRow = tiledlayout(tl, 1, nCols);
    tlRow.Layout.Tile = ii + 1;
    [~, ~, tl2, tl3] = plotBrain('lh', {surface, medialMask, mapsToPlot}, 'parent', tlRow, ...
        'view', {'ll', 'lm'}, 'tiledLayoutOptions', {1, nCols, 'TileSpacing', 'none'}, ...
        'tiledLayout2Options', {2, 1, 'TileSpacing', 'none'}, 'groupBy', 'data');
    tl2.Layout.Tile = 1;
    tl2.Layout.TileSpan = [1, nCols];
    title(tl2, {'Simulated activity on heterogeneous modes'; sprintf('hetero: %s | alpha: %.1f', ...
        heteroLabel, alphaVals(ii))})
    % Set colormap and label for c2 map
    c2_axs = tl3{1}.Children;
    colormap(c2_axs(1), viridis); colormap(c2_axs(2), viridis)
    xlabel(tl3{1}, 'c_s^2 map')
    % Plot time labels
    for jj = 1:length(tInterest)
        xlabel(tl3{jj+1}, sprintf('%.1f ms', tInterest(jj)))
    end
end

% Save figure
savecf(sprintf("%s/results/sim/hetero-%s_surf-%s_scale-%s_alpha-%.1f-%.1f_simActivity", ...
    projDir, heteroLabel, surf, scale, alphaVals(1), alphaVals(end)), ".png", 200)

%% Plot video of simulated activity

% Video of activity every 0.5 ms (increase this to better see the waves) t = param.T;
tInterest = 0:0.5:param.tmax;
is_time_ms = 1;
with_medial = 0;
cmap = parula;
show_colorbar = 1;
output_filename = sprintf("%s/results/sim/hetero-%s_surf-%s_scale-%s_alpha-%.1f-%.1f_simActivityVid", ...
    projDir, heteroLabel, surf, scale, alphaVals(1), alphaVals(end));
save_video = 1;

fig = video_surface_activity(surface, heteroSimActivity(:, :, 1), 'lh', param.T, ...
                             tInterest, is_time_ms, medialInds, with_medial, ...
                             cmap, show_colorbar, output_filename, save_video);

%% View results from Pang2023
data_results_folder = fullfile(BEdir, "data", "results");
model_wave_visual = load(sprintf('%s/model_wave_neural_visual_%s.mat', data_results_folder, hemisphere), ...
                                 'tspan', 'simulated_neural_visual_vertex', 'simulated_neural_visual_parcel', ...
                                 'ROIS', 'ROI_names');
figure; imagesc(model_wave_visual.simulated_neural_visual_vertex)


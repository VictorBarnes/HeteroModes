%% Simulate resting state FC using NFT wave equation
clear
clc

% Setup project by loading necessary functions
setupProject

% Load surface and mode parameters
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

% Set parameters for heterogeneous modes
modeParams_default = struct('heteroLabel', 'myelinmap', 'scale', 'cmean', 'alpha', 1.0, 'beta', 1.0);
heteroModesParams = [struct('beta', -1.0), struct('beta', -0.5), struct('beta', 0.5), struct('beta', 1.0)];
% heteroModesParams = [struct('alpha', 0.2), struct('alpha', 0.4), struct('alpha', 0.6), struct('alpha', 0.8), struct('alpha', 1.0)];
nHeteroBSs = length(heteroModesParams);     % Number of heterogeneous basis sets

disp("Loading modes and empirical data...")
% Load surface file
[vertices, faces] = read_vtk(sprintf('%s/atlas-yeo_space-%s_den-%s_surf-%s_hemi-%s_surface.vtk', ...
    surfDir, space, den, surf, hemi));
surface.vertices = vertices';
surface.faces = faces';
% Get cortex indices
medialMask = dlmread(sprintf('%s/atlas-yeo_space-%s_den-%s_hemi-%s_medialMask.txt', surfDir, space, ...
    den, hemi));
cortexInds = find(medialMask);

% Load parcellation
parcName = 'Glasser360';
parc = dlmread(sprintf('%s/data/parcellations/fsLR_32k_%s-lh.txt', BEdir, parcName));
nParcels = length(unique(parc(parc>0)));

% Load homogeneous eigenmodes and eigenvalues
homoDesc = 'hetero-%s_atlas-%s_space-%s_den-%s_surf-%s_hemi-%s_n-%i_scale-%s_maskMed-True';
% homoModes = dlmread('/home/vbarnes/kg98_scratch/jamesp/eigenmode_templates/fsLR_32k_midthickness-lh_evec_500.txt');
% homoEvals = dlmread('/home/vbarnes/kg98_scratch/jamesp/eigenmode_templates/fsLR_32k_midthickness-lh_eval_500.txt');
homoModes = dlmread(fullfile(emodeDir, sprintf(homoDesc, "None", atlas, space, den, surf, hemi, ...
    nModes, modeParams_default.scale) + "_emodes.txt"));
homoEvals = dlmread(fullfile(emodeDir, sprintf(homoDesc, "None", atlas, space, den, surf, hemi, ...
    nModes, modeParams_default.scale) + "_evals.txt"));

% Load heterogeneous eigenmodes and eigenvalues
heteroDesc = 'hetero-%s_atlas-%s_space-%s_den-%s_surf-%s_hemi-%s_n-%i_scale-%s_alpha-%.1f_beta-%.1f_maskMed-True';
heteroModes = zeros([size(homoModes), nHeteroBSs]);
heteroEvals = zeros(nHeteroBSs, nModes);
for ii=1:nHeteroBSs
% Extract current parameter values from the struct
    currentParams = modeParams_default;
    paramNames = fieldnames(modeParams_default);
    
    % Set default values for parameters not specified
    for jj=1:length(paramNames)
        if isfield(heteroModesParams(ii), paramNames{jj})
            currentParams.(paramNames{jj}) = heteroModesParams(ii).(paramNames{jj});
        end
    end

    % Load data
    heteroModes(:, :, ii) = dlmread(fullfile(emodeDir, sprintf(heteroDesc, currentParams.heteroLabel, atlas, ...
        space, den, surf, hemi, nModes, currentParams.scale, currentParams.alpha, currentParams.beta) + "_emodes.txt")); 
    heteroEvals(ii, :) = dlmread(fullfile(emodeDir, sprintf(heteroDesc, currentParams.heteroLabel, atlas, space, ...
        den, surf, hemi, nModes, currentParams.scale, currentParams.alpha, currentParams.beta) + "_evals.txt")); 
end

% Load empirical FC data
data = matfile(fullfile(BEdir, 'data', 'results', 'model_results_Glasser360_lh.mat'));
empFC = data.FC_emp;
% Calculate upper triangle indices (without diagonal values)
triuInds = find(triu(ones(size(empFC, 1), size(empFC, 2), 1), 1));

%% Set simulation parameters

waveParams = loadParameters_wave_func;
waveParams.tstep = 0.09; % in s
tpre =  50;                                     % burn time to remove transient
tpost = 863.2800;                               % steady-state time
waveParams.tmax = tpre + waveParams.tstep + tpost;
% param.tmax = 863.2800;  % in s
waveParams.tspan = [0, waveParams.tmax];
waveParams.T = 0:waveParams.tstep:waveParams.tmax;

waveParams.is_time_ms = 0;
method = 'Fourier';

waveParams.r_s = 28.9;      % (default) in mm
waveParams.gamma_s = 116;   % (default) in s^-1
if waveParams.is_time_ms==1
    waveParams.gamma_s = 116*1e-3;
end

balloonParams = loadParameters_balloon_func;
balloonParams.tstep = waveParams.tstep; % in s
balloonParams.tmax = waveParams.tmax;  % in s
balloonParams.tspan = waveParams.tspan;
balloonParams.T = waveParams.T;

% just to match HCP data's time length and TR
T = 1200;
TR = 0.72;       % in s

% random external input (to mimic resting state)
rand_ind = 1;
rng(rand_ind)
ext_input = randn(size(homoModes,1), length(waveParams.T));

%% Simulate FC using homogeneous modes

disp('Simulating FC using homoegeneous modes')
% simulate neural activity
[~, homoSimNeural] = model_neural_waves(homoModes, homoEvals, ext_input, waveParams, method);
% simulate BOLD activity from the neural activity
[~, homoSimBOLD] = model_BOLD_balloon(homoModes, homoSimNeural, balloonParams, method);

%%% Calculate FC of simulated BOLD data
time_steady_ind = dsearchn(waveParams.T', tpre);    % index of start of steady state
homoSimBOLD = homoSimBOLD(:,time_steady_ind:end);                        
% T_steady = (0:size(homoSimBOLD,2)-1)*param.tstep;
% downsample time series to match TR
homoSimBOLD = downsample(homoSimBOLD', floor(TR/(balloonParams.tstep)))';
% parcellate BOLD-fMRI time series
homoSimBOLD_parc = calc_parcellate(parc, homoSimBOLD);
% construct FC matrix (detrending the signal first)
homoSimBOLD_parc = detrend(homoSimBOLD_parc', 'constant');
homoSimBOLD_parc = homoSimBOLD_parc./repmat(std(homoSimBOLD_parc),T,1);
homoSimBOLD_parc(isnan(homoSimBOLD_parc)) = 0;
homoFC = homoSimBOLD_parc'*homoSimBOLD_parc/T;

%%% Calculate FC of simulated neural data
% parcellate BOLD-fMRI time series
homoSimNeural_parc = calc_parcellate(parc, homoSimNeural);
% construct FC matrix (detrending the signal first)
homoSimNeural_parc = detrend(homoSimNeural_parc', 'constant');
homoSimNeural_parc = homoSimNeural_parc./repmat(std(homoSimNeural_parc),10149,1);
homoSimNeural_parc(isnan(homoSimNeural_parc)) = 0;
homoNeuralFC = homoSimNeural_parc'*homoSimNeural_parc/10149;

%% Simulate FC using heterogeneous modes

disp('Simulating FC using heterogeneous modes')
heteroFCs = zeros(nParcels, nParcels, nHeteroBSs);
heteroNeuralFCs = zeros(nParcels, nParcels, nHeteroBSs);
for ii = 1:nHeteroBSs
    % simulate neural activity
    [~, heteroSimNeural] = model_neural_waves(heteroModes(:, :, ii), heteroEvals(ii, :), ext_input, waveParams, method);
    % simulate BOLD activity from the neural activity
    [~, heteroSimBOLD] = model_BOLD_balloon(heteroModes(:, :, ii), heteroSimNeural, balloonParams, method);
    
    %%% Calculate FC of simulated BOLD data
    time_steady_ind = dsearchn(waveParams.T', tpre);    % index of start of steady state
    heteroSimBOLD = heteroSimBOLD(:,time_steady_ind:end);                        
    % T_steady = (0:size(heteroSimBOLD,2)-1)*param.tstep;
    % downsample time series to match TR
    heteroSimBOLD = downsample(heteroSimBOLD', floor(TR/(balloonParams.tstep)))';
    % parcellate BOLD-fMRI time series
    heteroSimBOLD_parc = calc_parcellate(parc, heteroSimBOLD);
    % construct FC matrix (detrending the signal first)
    heteroSimBOLD_parc = detrend(heteroSimBOLD_parc', 'constant');
    heteroSimBOLD_parc = heteroSimBOLD_parc./repmat(std(heteroSimBOLD_parc),T,1);
    heteroSimBOLD_parc(isnan(heteroSimBOLD_parc)) = 0;
    heteroFCs(:, :, ii) = heteroSimBOLD_parc'*heteroSimBOLD_parc/T;

    %%% Calculate FC of simulated neural data
    % parcellate BOLD-fMRI time series
    heteroSimNeural_parc = calc_parcellate(parc, heteroSimNeural);
    % construct FC matrix (detrending the signal first)
    heteroSimNeural_parc = detrend(heteroSimNeural_parc', 'constant');
    heteroSimNeural_parc = heteroSimNeural_parc./repmat(std(heteroSimNeural_parc),10149,1);
    heteroSimNeural_parc(isnan(heteroSimNeural_parc)) = 0;
    heteroNeuralFCs(:, :, ii) = heteroSimNeural_parc'*heteroSimNeural_parc/10149;    
end

%% Plot results

% Calculate Node FC of empirical data (mean of each row excluding the diagonals)
empNodeFC = mean(empFC - diag(diag(empFC)), 2);

figure('Position', [100, 100, 350*(nHeteroBSs+1), 1200]);
tl1 = tiledlayout(1, 1 + nHeteroBSs);

%%% Plot homogeneous model results
tl2 = tiledlayout(tl1, 4, 1, 'TileSpacing', 'tight');
tl2.Layout.Tile = 1;
title(tl2, 'Homogeneous model');
% Plot emprical and model FC matrices
nexttile(tl2); imagesc(empFC); colormap(bluewhitered); colorbar; title('Empirical FC');
nexttile(tl2); imagesc(homoFC); colormap(bluewhitered); colorbar; 
title('Simulated FC');
% Calculate edge FC
nexttile(tl2);
scatter(homoFC(triuInds), empFC(triuInds), 16, 'filled');
homoEdgeFC = corr(homoFC(triuInds), empFC(triuInds));
title(sprintf('Edge FC (r = %.2f)', homoEdgeFC)); xlabel('model'); ylabel('empirical');
% Calculate Node FC (mean of each row excluding the diagonals)
homoNodeFC = mean(homoFC - diag(diag(homoFC)), 2);
homoNodeFCcorr = corr(homoNodeFC, empNodeFC);
nexttile(tl2);
scatter(homoNodeFC, empNodeFC, 16, 'filled');
title(sprintf('Node FC (r = %.2f)', homoNodeFCcorr)); xlabel('model'); ylabel('empirical')

%%% Plot heterogeneous model results
for ii = 1:nHeteroBSs
    heteroFC = heteroFCs(:, :, ii);
    tl2 = tiledlayout(tl1, 4, 1, 'TileSpacing', 'tight');
    tl2.Layout.Tile = ii + 1;
    field = fieldnames(heteroModesParams(ii));
    value = heteroModesParams(ii).(field{1});
    title(tl2, sprintf('Heterogeneous model (%s: %.1f)', field{1}, value))

    % Plot emprical and model FC matrices
    nexttile(tl2); imagesc(empFC); colormap(bluewhitered); colorbar; title('Empirical FC');
    nexttile(tl2); imagesc(heteroFC); colormap(bluewhitered); colorbar; 
    title('Simulated FC');
    % Calculate and plot edge FC
    nexttile(tl2); scatter(heteroFC(triuInds), empFC(triuInds), 16, 'filled');
    heteroEdgeFC = corr(heteroFC(triuInds), empFC(triuInds));
    title(sprintf('Edge FC (r = %.2f)', heteroEdgeFC)); xlabel('model'); ylabel('empirical');
    % Calculate and plot Node FC (mean of each row excluding the diagonals)
    heteroNodeFC = mean(heteroFC - diag(diag(heteroFC)), 2);
    heteroNodeFCcorr = corr(heteroNodeFC, empNodeFC);
    nexttile(tl2); scatter(heteroNodeFC, empNodeFC, 16, 'filled');
    title(sprintf('Node FC (r = %.2f)', heteroNodeFCcorr)); xlabel('model'); ylabel('empirical')
end

% savecf(sprintf("%s/simulateFC/hetero-%s_scale-%s_alpha-%.1f_beta-%.1f_reconAccuracy", ...
%     resultsDir, heteroLabel, surf, scale, alphaVals(ii), beta), ".png", 150)
      
%% Plot FC of simulated neural data

figure('Position', [100, 100, 1800, 400]);
tl1 = tiledlayout(1, 4);
for ii=1:4
    nexttile
    imagesc(heteroNeuralFCs(:, :, ii)); colormap(bluewhitered); colorbar;
    
    field = fieldnames(heteroModesParams(ii));
    value = heteroModesParams(ii).(field{1});
    title(sprintf('%s: %.1f', field{1}, value))
end


%% Plot homo vs hetero edge FC
figure('Position', [100, 100, 1800, 400]);
tl1 = tiledlayout(1, 4);
for ii=1:4
    nexttile
    scatter(homoFC, heteroFCs(:, :, ii))
    
    field = fieldnames(heteroModesParams(ii));
    value = heteroModesParams(ii).(field{1});
    title(sprintf('%s: %.1f', field{1}, value))
    xlabel('Homogeneous'); ylabel('Heterogeneous')
end


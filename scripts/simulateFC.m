%% Simulate resting state FC using NFT wave equation
clear
clc

% Setup project by loading necessary functions
setupProject

% Load surface and mode parameters
config = jsondecode(fileread(fullfile(pwd, "config.json")));
emodeDir = config.emode_dir;
surfDir = config.surface_dir;
resultsDir = config.results_dir;
reposDir = config.repos_dir;
BEdir = fullfile(reposDir, 'BrainEigenmodes');
atlas = config.atlas;
space = config.space;
den = config.den;
surf = config.surf;
hemi = config.hemi;
nModes = config.n_modes;
heteroLabel = config.hetero_label;
alphaVals = config.alpha_vals;
betaVals = config.beta_vals;

% Get all alpha and beta combinations
[A, B] = meshgrid(alphaVals, betaVals);
abCombs = reshape(cat(2,A',B'), [], 2);
nBasisSets = 1 + size(abCombs, 1);     % Number of homogeneous and heterogeneous basis sets

% Load surface file
[vertices, faces] = read_vtk(sprintf('%s/atlas-%s_space-%s_den-%s_surf-%s_hemi-%s_surface.vtk', ...
    surfDir, atlas, space, den, surf, hemi));
surface.vertices = vertices';
surface.faces = faces';
% Get cortex indices
medialMask = dlmread(sprintf('%s/atlas-%s_space-%s_den-%s_hemi-%s_medialMask.txt', surfDir, atlas, ...
    space, den, hemi));
cortexInds = find(medialMask);

% Load parcellation
parcName = 'Glasser360';
parc = dlmread(sprintf('%s/data/parcellations/fsLR_32k_%s-lh.txt', BEdir, parcName));
nParcels = length(unique(parc(parc>0)));

% Load eigenmodes and eigenvalues
fprintf("Loading %i basis sets... ", nBasisSets);
desc = 'hetero-%s_atlas-%s_space-%s_den-%s_surf-%s_hemi-%s_n-%i_alpha-%.1f_beta-%.1f_maskMed-True';
basisSets_modes = zeros([size(surface.vertices, 1), nModes, nBasisSets]);
basisSets_evals = zeros(nModes, nBasisSets);
for ii=1:nBasisSets
    % The first basis set to load is the homogeneous baisis set. All others are heterogeneous
    if ii == 1
        heteroLabel = "None";
        alpha = 0.0;
        beta = 0.0;
    else
        heteroLabel = config.hetero_label;
        alpha = abCombs(ii-1, 1);
        beta = abCombs(ii-1, 2);
    end

    % Load data
    basisSets_modes(:, :, ii) = dlmread(fullfile(emodeDir, sprintf(desc, heteroLabel, atlas, ...
        space, den, surf, hemi, nModes, alpha, beta) + "_emodes.txt")); 
    basisSets_evals(:, ii) = dlmread(fullfile(emodeDir, sprintf(desc, heteroLabel, atlas, space, ...
        den, surf, hemi, nModes, alpha, beta) + "_evals.txt")); 
end
fprintf("done\n");

% Load empirical FC data
data = matfile(fullfile(BEdir, 'data', 'results', 'model_results_Glasser360_lh.mat'));
empFC = data.FC_emp;

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
extInput = randn(size(surface.vertices, 1), length(waveParams.T));

%% Simulate FC using homogeneous and heterogeneous modes

fprintf('Simulating FC using %i different basis sets... ', nBasisSets)
simFCmatrices_bold = zeros(nParcels, nParcels, nBasisSets);
simFCmatrices_neural = zeros(nParcels, nParcels, nBasisSets);
for ii = 1:nBasisSets
    % simulate neural activity
    [~, neuralSim] = model_neural_waves(basisSets_modes(:, :, ii), basisSets_evals(:, ii), extInput, waveParams, method);
    % simulate BOLD activity from the neural activity
    [~, boldSim] = model_BOLD_balloon(basisSets_modes(:, :, ii), neuralSim, balloonParams, method);
    
    %%% Calculate FC of simulated BOLD data
    time_steady_ind = dsearchn(waveParams.T', tpre);    % index of start of steady state
    boldSim = boldSim(:,time_steady_ind:end);                        
    % T_steady = (0:size(heteroSimBOLD,2)-1)*param.tstep;
    % downsample time series to match TR
    boldSim = downsample(boldSim', floor(TR/(balloonParams.tstep)))';
    % parcellate BOLD-fMRI time series
    boldSim_parc = calc_parcellate(parc, boldSim);
    % construct FC matrix (detrending the signal first)
    boldSim_parc = detrend(boldSim_parc', 'constant');
    boldSim_parc = boldSim_parc./repmat(std(boldSim_parc),T,1);
    boldSim_parc(isnan(boldSim_parc)) = 0;
    simFCmatrices_bold(:, :, ii) = boldSim_parc'*boldSim_parc/T;

    %%% Calculate FC of simulated neural data
    % parcellate BOLD-fMRI time series
    neuralSim_parc = calc_parcellate(parc, neuralSim);
    % construct FC matrix (detrending the signal first)
    neuralSim_parc = detrend(neuralSim_parc', 'constant');
    neuralSim_parc = neuralSim_parc./repmat(std(neuralSim_parc),size(neuralSim, 2),1);
    neuralSim_parc(isnan(neuralSim_parc)) = 0;
    simFCmatrices_neural(:, :, ii) = neuralSim_parc'*neuralSim_parc/size(neuralSim, 2);    
end
fprintf('done\n')

%% Plot results

% Calculate upper triangle indices (without diagonal values)
triuInds = find(triu(ones(size(empFC, 1), size(empFC, 2), 1), 1));
% Calculate Node FC of empirical data (mean of each row excluding the diagonals)
empNodeFC = mean(empFC - diag(diag(empFC)), 2);

figure('Position', [100, 100, 350*(nBasisSets), 900]);
tl1 = tiledlayout(1, nBasisSets);
title(tl1, {'Simulating FC using wave model'; sprintf('(hetero: %s)', heteroLabel)}, 'FontSize', 16)

for ii = 1:nBasisSets
    % The first basis set to load is the homogeneous baisis set. All others are heterogeneous       
    simFC = simFCmatrices_bold(:, :, ii);
    tl2 = tiledlayout(tl1, 4, 1, 'TileSpacing', 'tight');
    tl2.Layout.Tile = ii;
    if ii == 1
        title(tl2, 'Homogeneous model');
    else
        title(tl2, {'Heterogeneous model', sprintf('(alpha: %.1f | beta: %.1f)', abCombs(ii-1, 1), ...
            abCombs(ii-1, 2))})
    end

    % Plot emprical and model FC matrices
    nexttile(tl2); imagesc(empFC); colormap(bluewhitered_mg); colorbar; title('Empirical FC');
    nexttile(tl2); imagesc(simFC); colormap(bluewhitered_mg); 
    cb = colorbar; cb.Ruler.TickLabelFormat = '%.1f';
    title('Simulated FC');
    % Calculate and plot edge FC
    nexttile(tl2); scatter(simFC(triuInds), empFC(triuInds), 16, 'filled');
    edgeFC = corr(simFC(triuInds), empFC(triuInds));
    title(sprintf('Edge FC (r = %.2f)', edgeFC)); xlabel('model'); ylabel('empirical');
    % Calculate and plot Node FC (mean of each row excluding the diagonals)
    nodeFC = mean(simFC - diag(diag(simFC)), 2);
    nodeFCcorr = corr(nodeFC, empNodeFC);
    nexttile(tl2); scatter(nodeFC, empNodeFC, 16, 'filled');
    title(sprintf('Node FC (r = %.2f)', nodeFCcorr)); xlabel('model'); ylabel('empirical')
end

% savecf(sprintf("%s/simulateFC/hetero-%s_alpha-%.1f_beta-%.1f_reconAccuracy", ...
%     resultsDir, heteroLabel, surf, alphaVals(ii), beta), ".png", 150)



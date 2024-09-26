clear
clc

setupProject

emodes1 = readmatrix(fullfile(projDir, 'data', 'eigenmodes', 'atlas-S1200_space-fsLR_den-32k_surf-midthickness_hemi-L_maskMed-True_n-500_emodes.txt'));
evals1 = readmatrix(fullfile(projDir, 'data', 'eigenmodes', 'atlas-S1200_space-fsLR_den-32k_surf-midthickness_hemi-L_maskMed-True_n-500_evals.txt'));
[nVerts_orig, nModes] = size(emodes1);
medMask = readmatrix(fullfile(projDir, 'data', 'surfaces', 'atlas-S1200_space-fsLR_den-32k_hemi-L_medialMask.txt'));
cortexInds = find(medMask);
medialInds = find(~medMask);
nVerts_mask = length(cortexInds);

[vertices, faces] = read_vtk('/home/vbarnes/kg98/vbarnes/HeteroModes/data/surfaces/atlas-S1200_space-fsLR_den-32k_surf-midthickness_hemi-L_surface.vtk');
surface.vertices = vertices';
surface.faces = faces';

% Set first mode to be constant
emodes1(cortexInds, 1) = mean(emodes1(cortexInds, 1), 'all');
% Load B matrix
disp('Loading B matrix...')
B = load(fullfile(projDir, 'data', 'Bmatrix.mat')).B;


%% Set wave model parameters

fprintf('Setting simulating parameters... '); tic
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
method = 'Fourier';
% method = 'Fourier';

param.r_s = 30;      % (default) in mm
param.gamma_s = 116; % (default) in s^-1
if param.is_time_ms==1
    param.gamma_s = 116*1e-3;
end    

fprintf('done. '); toc; fprintf('\n')

%% Run wave model
rng(1)
extInput = readmatrix('/home/vbarnes/kg98/vbarnes/HeteroModes/data/white_noise.txt');
% simulate neural activity
[~, simNeural] = model_neural_waves(emodes1(cortexInds, :), evals1, extInput(cortexInds, :), param, method);

simNeural_full = nan(nVerts_orig, size(simNeural, 2));
simNeural_full(cortexInds, :) = simNeural;

%% Plot video of simulated activity

% Video of activity every 0.5 ms (increase this to better see the waves) t = param.T;
tInterest = 0:0.5:param.tmax;
with_medial = 0;
cmap = parula;
show_colorbar = 1;
output_filename = sprintf("%s/results/sim/simActivityVid", projDir);
save_video = 0;

fig = video_surface_activity(surface, simNeural_full, 'lh', param.T, ...
                             tInterest, param.is_time_ms, medialInds, with_medial, ...
                             cmap, show_colorbar, output_filename, save_video);
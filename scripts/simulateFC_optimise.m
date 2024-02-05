%% Optimise alpha and beta parameters by simulating resting state FC for a range of values of each
clear
clc

% Setup project by loading necessary functions
setupProject

% Load surface and mode parameters
config = jsondecode(fileread(fullfile(pwd, "config.json")));
emodeDir = config.emode_dir;
surfDir = config.surface_dir;
projDir = config.project_dir;
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
nRuns = 50;     % Number of randomly seeded runs to do
fprintf('Hetero: %s | Runs: %i\n', heteroLabel, nRuns)
fprintf(['alpha: ' repmat(' %.1f ',1,numel(alphaVals)) '\n'], alphaVals);
fprintf(['beta: ' repmat(' %.1f ',1,numel(betaVals)) '\n'], betaVals);

% Get all alpha and beta combinations
[A, B] = meshgrid(alphaVals, betaVals);
alphaBetaCombs = reshape(cat(2,A',B'), [], 2);
% Initial row is for the homogeneous basis set (alpha and beta are both 0)
alphaBetaCombs = [0, 0; alphaBetaCombs];    
nBasisSets = size(alphaBetaCombs, 1);     % Number of homogeneous and heterogeneous basis sets

% Load surface file
[vertices, faces] = read_vtk(sprintf('%s/atlas-%s_space-%s_den-%s_surf-%s_hemi-%s_surface.vtk', ...
    surfDir, atlas, space, den, surf, hemi));
surface.vertices = vertices';
surface.faces = faces';
nVertices = size(surface.vertices, 1);
% Get cortex indices
medialMask = readmatrix(sprintf('%s/atlas-%s_space-%s_den-%s_hemi-%s_medialMask.txt', surfDir, atlas, ...
    space, den, hemi));
cortexInds = find(medialMask);

% Load parcellation
parcName = 'Glasser360';
parc = readmatrix(sprintf('%s/data/parcellations/fsLR_32k_%s-lh.txt', BEdir, parcName));
nParcels = length(unique(parc(parc>0)));

% Load eigenmodes and eigenvalues
fprintf("Loading %i basis sets... ", nBasisSets); tic;
desc = 'hetero-%s_atlas-%s_space-%s_den-%s_surf-%s_hemi-%s_n-%i_alpha-%.1f_beta-%.1f_maskMed-True';
basisSets_modes = nan([size(surface.vertices, 1), nModes, nBasisSets]);
basisSets_evals = nan(nModes, nBasisSets);
for ii=1:nBasisSets
    % The first basis set to load is the homogeneous baisis set. All others are heterogeneous
    if ii == 1
        heteroLabel = "None";
    else
        heteroLabel = config.hetero_label;
    end
    alpha = alphaBetaCombs(ii, 1);
    beta = alphaBetaCombs(ii, 2);

    % Load data
    basisSets_modes(:, :, ii) = readmatrix(fullfile(emodeDir, sprintf(desc, heteroLabel, atlas, ...
        space, den, surf, hemi, nModes, alpha, beta) + "_emodes.txt")); 
    basisSets_evals(:, ii) = readmatrix(fullfile(emodeDir, sprintf(desc, heteroLabel, atlas, space, ...
        den, surf, hemi, nModes, alpha, beta) + "_evals.txt")); 
end
fprintf("done. "); toc; fprintf('\n')

% Load empirical FC data
empData = load(fullfile(projDir, 'data', 'BOLD_empirical_HCP_S255_Glasser360-lh.mat'));
participantIDs = empData.subject_list(1:50);    % Only need this for reference   
empBOLD = empData.BOLD(:, :, 1:50);    % TEMP: only use first 50 subjects for now
nSubjects = size(empBOLD, 3);

% TODO: Separate training and testing datasets

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

% index of start of steady state
time_steady_ind = dsearchn(waveParams.T', tpre);    

%% Compute FC and evaluation metrics of empirical data 

empFCs = nan(nParcels, nParcels, nSubjects);
empFCDs = nan(694431, nSubjects);       % TODO: make the first dim generalisable
for ii=1:nSubjects
    empBOLD_current = empBOLD(:, :, ii);                        
    % construct FC matrix (detrending the signal first)
    empBOLD_current = detrend(empBOLD_current', 'constant');
    empBOLD_current = empBOLD_current./repmat(std(empBOLD_current),T,1);
    empBOLD_current(isnan(empBOLD_current)) = 0;
    empFCs(:, :, ii) = empBOLD_current'*empBOLD_current/T;

    % Compute FCD for each empirical FC
    empFCDs(:, ii) = calc_phase_fcd(empBOLD_current, TR);
end
% Average empirical FC over all subjects (for edge and node FC evaluation)
empFC_avg = mean(empFCs, 3);

% Calculate Node FC of empirical data (mean of each row excluding the diagonals)
empNodeFC = mean(empFC_avg - diag(diag(empFC_avg)), 2);

%% Simulate FC using homogeneous and heterogeneous modes and compute evaluation metrics

% Calculate upper triangle indices (without diagonal values)
triuInds = find(triu(ones(nParcels, nParcels), 1));

% Initialise matrices for evaluation metrics
simFCs_avg = nan(nParcels, nParcels, nBasisSets);   % Simulated FCs for each basis set averaged over all runs
simEdgeFCs = nan(1, nBasisSets);
simNodeFCs = nan(1, nBasisSets);
simKSFCDs = nan(1, nBasisSets);
simFCDs_avg = nan(694431, nBasisSets);
fprintf('Simulating FC using %i different basis sets... \n', nBasisSets); tic;
parfor ii=1:nBasisSets
    % Initialise matrix for simulated data
    simFCs = nan(nParcels, nParcels, nRuns);
    simFCDs = nan(694431, nRuns)
    for run=1:nRuns
        % random external input (to mimic resting state)
        rng(run)
        extInput = randn(nVertices, length(waveParams.T));
        
        % simulate neural activity
        [~, simNeural] = model_neural_waves(basisSets_modes(:, :, ii), basisSets_evals(:, ii), extInput, waveParams, method);
        % simulate BOLD activity from the neural activity
        [~, simBOLD] = model_BOLD_balloon(basisSets_modes(:, :, ii), simNeural, balloonParams, method);
        
        % Calculate FC of simulated BOLD data
        simBOLD = simBOLD(:,time_steady_ind:end);                        
        % downsample time series to match TR
        simBOLD = downsample(simBOLD', floor(TR/(balloonParams.tstep)))';
        % parcellate BOLD-fMRI time series
        simBOLD_parc = calc_parcellate(parc, simBOLD);
        % construct FC matrix (detrending the signal first)
        simBOLD_parc = detrend(simBOLD_parc', 'constant');
        simBOLD_parc = simBOLD_parc./repmat(std(simBOLD_parc),T,1);
        simBOLD_parc(isnan(simBOLD_parc)) = 0;
        simFCs(:, :, run) = simBOLD_parc'*simBOLD_parc/T;

        % Compute FCD for each simulated FC
        simFCDs(:, run) = calc_phase_fcd(simBOLD_parc, TR);
    end
    
    % Average simulated FC over all runs
    simFC_avg = mean(simFCs, 3);
    simFCs_avg(:, :, ii) = simFC_avg;
    
    % Compute edge FC
    simEdgeFCs(ii) = corr(simFC_avg(triuInds), empFC_avg(triuInds), 'rows', 'complete');
    % Compute Node FC (mean of each row excluding the diagonals)
    simNodeFC = mean(simFC_avg - diag(diag(simFC_avg)), 2);
    simNodeFCs(ii) = corr(simNodeFC, empNodeFC, 'rows', 'complete');

    % Compute KS statistic between empirical FCDs and simulated FCDs
    [~, ~, simKSFCDs(ii)] = kstest2(empFCDs(:), simFCDs(:));
    % Compute and store FCD averaged over runs
    simFCDs_avg(:, ii) = mean(simFCDs, 2);

end
toc;

% Save results
fprintf("Saving results... ")
outputFolder = fullfile(projDir, 'results', 'simulateFC', 'optimise');
outputDesc = 'hetero-%s_empDset-hcp_nRuns-%i_simulateFCresults_50subjs_noCrossVal.mat';
save(fullfile(outputFolder, sprintf(outputDesc, heteroLabel, nRuns)), 'alphaBetaCombs', ...
    'simFCs_avg', 'simEdgeFCs', 'simNodeFCs', 'simKSFCDs', 'simFCDs_avg', 'empFCDs');
fprintf('done.\n')

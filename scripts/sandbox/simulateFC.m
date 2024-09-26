function [simFC_avg, edgeFCcorr, nodeFCcorr, FCDks] = simulateFC(emodes, evals, medialMask, nRuns, saveAll)
%% Simulate FC for given alpha and beta values then calculate evaluation metrics.
%
% Inputs: 
%       configFile : configuration file
%       heteroLabel : label of heterogeneous map that the modes have been paramaterized by (str)
%       alpha : alpha scaling value (float)
%       beta : beta scaling value (float)
%       nRuns : number of model FC simulations to run
%       saveAll : whether to save all FC output data or just the evaluation metrics
%
% Outputs:
%       simFC_avg : simulated FC averaged over runs
%       edgeFCcorr : edge FC correlation of average simulated FC with average empirical FC
%       nodeFCcorr : node FC correlation of average simulated FC with average empirical FC
%       FCDks : KS statistic taken between the FCD of the simulated data and the FCD of the
%           empirical data
%
% Original: Victor Barnes, Monash University, 2024

%% Load data

% Setup project by loading necessary functions
setupProject

% Load eigenmodes and eigenvalues
fprintf("Loading eigenmodes and eigenvalues... "); tic;
if ischar(emodes) || isStringScalar(emodes)
    emodes = readmatrix(emodes); 
elseif ~ismatrix(emodes)
    error('emodes must be a filepath or a matrix')
end
% Load eigenvalues
if ischar(evals) || isStringScalar(evals)
    evals = readmatrix(evals); 
elseif ~ismatrix(evals)
    error('evals must be a filepath or a matrix')
end
% Load medialMask
if ischar(medialMask) || isStringScalar(medialMask)
    medialMask = readmatrix(medialMask); 
elseif ~ismatrix(medialMask)
    error('medialMask must be a filepath or a matrix')
end
fprintf("done. "); toc; fprintf('\n')
cortexInds = find(medialMask);
nVertices = size(emodes(cortexInds), 1);

% TODO: add empBOLD as an input variable
% Load empirical BOLD data
empData = load(fullfile(projDir, 'data', 'BOLD_empirical_HCP_S255_Glasser360-lh.mat'));
empBOLD = empData.BOLD(:, :, 1:50);    % TEMP: only use first 50 subjects for now
nSubjects = size(empBOLD, 3);

% Load parcellation
parcName = 'Glasser360';
parc = readmatrix(sprintf('%s/data/parcellations/fsLR_32k_%s-lh.txt', ...
    fullfile(config.repos_dir, 'BrainEigenmodes'), parcName));
nParcels = length(unique(parc(parc>0)));

%% Set simulation parameters

fprintf('Setting simulating parameters... '); tic
waveParams = loadParameters_wave_func;
waveParams.tstep = 0.09; % in s
tpre =  50;                                     % burn time to remove transient
tpost = 863.2800;                               % time during steady-state
waveParams.tmax = tpre + waveParams.tstep + tpost;
waveParams.tspan = [0, waveParams.tmax];
waveParams.T = 0:waveParams.tstep:waveParams.tmax;

waveParams.is_time_ms = false;
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

fprintf('done. '); toc; fprintf('\n')

%% Compute FC and evaluation metrics of empirical data 
% TODO: compute empirical FC and metrics in a separate function
fprintf('Computing FC and evaluation metrics for empirical data... '); tic
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
fprintf('done. '); toc; fprintf('\n')

%% Simulate FC using homogeneous and heterogeneous modes and compute evaluation metrics

fprintf('Computing FC and evaluation metrics for simulated data (%i runs)... ', nRuns); tic
% Calculate upper triangle indices (without diagonal values)
triuInds = find(triu(ones(nParcels, nParcels), 1));

% Initialise matrix for simulated data
simFCs = nan(nParcels, nParcels, nRuns);
simFCDs = nan(694431, nRuns);
for run=1:nRuns
    % random external input (to mimic resting state)
    rng(run)
    extInput = randn(nVertices, length(waveParams.T));
    
    % simulate neural activity
    [~, simNeural] = model_neural_waves(emodes(cortexInds, :), evals, extInput, waveParams, method);
    % simulate BOLD activity from the neural activity
    [~, simBOLD] = model_BOLD_balloon(emodes(cortexInds, :), simNeural, balloonParams, method);
    
    % Calculate FC of simulated BOLD data
    simBOLD = simBOLD(:,time_steady_ind:end);                        
    % downsample time series to match TR
    simBOLD = downsample(simBOLD', floor(TR/(balloonParams.tstep)))';
    % parcellate BOLD-fMRI time series
    simBOLD_parc = calc_parcellate(parc(cortexInds), simBOLD);
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

% Compute edge FC
edgeFCcorr = corr(simFC_avg(triuInds), empFC_avg(triuInds), 'rows', 'complete');
% Compute Node FC (mean of each row excluding the diagonals)
simNodeFC = mean(simFC_avg - diag(diag(simFC_avg)), 2);
nodeFCcorr = corr(simNodeFC, empNodeFC, 'rows', 'complete');
% Compute KS statistic between empirical FCDs and simulated FCDs
[~, ~, FCDks] = kstest2(empFCDs(:), simFCDs(:));
fprintf('done. '); toc; fprintf('\n')

if saveAll
    fprintf('Saving ALL results... '); tic
    outputFolder = fullfile(config.project_dir, 'results', 'simulateFC', 'optimise');
    outputDesc = 'hetero-%s_alpha-%.1f_beta-%.1f_empDset-hcp_nRuns-%i_nSubj-50_crossVal-False_simulateFCresults_saveAll.mat';
    save(fullfile(outputFolder, sprintf(outputDesc, heteroLabel, alpha, beta, nRuns)), ...
        'edgeFCcorr', 'nodeFCcorr', 'FCDks', 'simFC_avg', 'empFC_avg', 'simFCDs', 'empFCDs');
else
    fprintf('Saving ONLY evaluation metric results... '); tic
    outputFolder = fullfile(config.project_dir, 'results', 'simulateFC', 'optimise', 'temp');
    outputDesc = 'hetero-%s_alpha-%.1f_beta-%.1f_empDset-hcp_nRuns-%i_nSubj-50_crossVal-False_simulateFCresults.mat';
    save(fullfile(outputFolder, sprintf(outputDesc, heteroLabel, alpha, beta, nRuns)), ...
        'edgeFCcorr', 'nodeFCcorr', 'FCDks');
end
fprintf('done. '); toc; fprintf('\n')

end
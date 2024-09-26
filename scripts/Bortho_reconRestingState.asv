% clear
% clc

setupProject

emodes_orig = readmatrix(fullfile(projDir, 'data', 'eigenmodes', 'hetero-None_atlas-S1200_space-fsLR_den-32k_surf-midthickness_hemi-L_n-500_maskMed-True_emodes.txt'));
emodes_orig = emodes_orig(:, 1:200);
[nVerts_orig, nModes] = size(emodes_orig);
medMask = readmatrix(fullfile(projDir, 'data', 'surfaces', 'atlas-S1200_space-fsLR_den-32k_hemi-L_medialMask.txt'));
cortexInds = find(medMask);
nVerts_mask = length(cortexInds);

% Set first mode to be constant
emodes_orig(cortexInds, 1) = mean(emodes_orig(cortexInds, 1), 'all');
% Load B matrix
disp('Loading B matrix...')
tic
B = load(fullfile(projDir, 'data', 'Bmatrix.mat')).B;
toc

%% Load parcellation and resting state data

parc = readmatrix(fullfile(projDir, 'data', 'fsLR_32k_Glasser360-lh.txt'));
nParcels = length(unique(parc(parc>0)));

% Load resting state data
data = gifti(char(fullfile(projDir, 'data', 'rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean_lh.func.gii')));
rsfMRI = data.cdata;
rsfMRI_masked = rsfMRI(cortexInds, :);
% Z-score data
rsfMRI_masked = zscore(rsfMRI_masked.').';

nT = size(rsfMRI, 2);

%% Reconstruct each time point

betaCoeffs_orig = cell(nModes, 1); betaCoeffs_ortho = cell(nModes, 1);
recon_orig = nan(nVerts_mask, nModes, nT); recon_ortho = nan(nVerts_mask, nModes, nT);
corrCoeffs_orig = nan(nModes, nT); corrCoeffs_ortho = nan(nModes, nT);
for n = 1:nModes
   betaCoeffs_orig{n} = (emodes_orig(cortexInds, 1:n).'*emodes_orig(cortexInds, 1:n))\(emodes_orig(cortexInds, 1:n).'*rsfMRI_masked);
%    betaCoeffs_orig{n} = calc_eigendecomposition(rsfMRI(cortexInds, :), emodes_orig(cortexInds, 1:n), 'matrix');
   recon_orig(:, n, :) = emodes_orig(cortexInds, 1:n) * betaCoeffs_orig{n};
   corrCoeffs_orig(n,:) = diag(corr(rsfMRI_masked, squeeze(recon_orig(:, n, :))));

    betaCoeffs_ortho{n} = emodes_orig(cortexInds, 1:n).' * B(cortexInds, cortexInds) * rsfMRI_masked;
%    betaCoeffs_ortho{n} = calc_eigendecomposition(rsfMRI(cortexInds, :), emodes_norm(cortexInds, 1:n), 'matrix');
   recon_ortho(:, n, :) = emodes_orig(cortexInds, 1:n) * betaCoeffs_ortho{n};
   corrCoeffs_ortho(n,:) = diag(corr(rsfMRI_masked, squeeze(recon_ortho(:, n, :))));
end

%% Plot recon of empirical data, orig method, and Bortho method for 4 random vertices 
rng(1)
vertIDs = randi([1, 29696], 1, 4);

figure;
for ii = 1:length(vertIDs)
    nexttile()
    plot(rsfMRI_masked(vertIDs(ii), 1:200), 'DisplayName', 'Empirical');
    hold on; 
    plot(squeeze(recon_orig(vertIDs(ii), 200, 1:200)), 'DisplayName', 'Original'); 
    plot(squeeze(recon_ortho(vertIDs(ii), 200, 1:200)), 'DisplayName', 'Bortho');
    legend('Location', 'southeast')
    title(gca, sprintf('Vertex Id: %i', vertIDs(ii)))
end



%% Plot B matrix

figure;

cmap = jet;
s = 0.5;
[ii, jj, Bnnz] = find(B);
scatter(1, 1, s, 0)
hold on
scatter(ii, jj, s, Bnnz);
set(gca, 'color', cmap(1, :), 'ColorScale', 'log')
colorbar

clear
clc

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

%% Load empirical data
dset = "nm";

disp("Loading data...")
if dset == "hcp"
    % Load hcp task fMRI data
    hcpData = load(fullfile(projDir, 'data', 'S255_tfMRI_ALLTASKS_raw_lh.mat'));
    hcpData = hcpData.zstat;
    % Only load first subject for each task contrast
    empiricalLabels = fieldnames(hcpData);
    empiricalData = nan(size(hcpData.(empiricalLabels{1}), 1), numel(empiricalLabels));
    for ii=1:numel(empiricalLabels)
        % Average across all subjects for each hcp contrast
        empiricalData(:, ii) = mean(hcpData.(empiricalLabels{ii}), 2, "omitnan");
    end
elseif dset == "nm" 
    nmData = load(fullfile(projDir, 'data', 'neuromaps_noPET.mat'));
    empiricalLabels = fieldnames(nmData);

    empiricalData = nan(nVerts_orig, numel(empiricalLabels));
    for ii = 1:numel(empiricalLabels)
        empiricalData(:, ii) = nmData.(empiricalLabels{ii});
    end

    % Remove devexp (id: 11) and genepc1 (id: 16) which have nans
    empiricalData(:, [11, 16]) = [];
    empiricalLabels([11, 16]) = [];
end
nImages = size(empiricalData, 2);

%% Calculate recon beta coefficients using 1 to num_modes eigenmodes
disp('Calculating eigenreconstruction...')

betaCoeffs_orig = cell(nModes, 1); betaCoeffs_ortho = cell(nModes, 1);
recon_orig = nan(nVerts_mask, nModes, nImages); recon_ortho = nan(nVerts_mask, nModes, nImages);
corrCoeffs_orig = nan(nModes, nImages); corrCoeffs_ortho = nan(nModes, nImages);
for n = 1:nModes
   betaCoeffs_orig{n} = (emodes_orig(cortexInds, 1:n).'*emodes_orig(cortexInds, 1:n))\(emodes_orig(cortexInds, 1:n).'*empiricalData(cortexInds, :));
   recon_orig(:, n, :) = emodes_orig(cortexInds, 1:n) * betaCoeffs_orig{n};
   corrCoeffs_orig(n,:) = diag(corr(empiricalData(cortexInds, :), squeeze(recon_orig(:, n, :))));

   betaCoeffs_ortho{n} = emodes_orig(cortexInds, 1:n).' * B(cortexInds, cortexInds) * empiricalData(cortexInds, :);
   recon_ortho(:, n, :) = emodes_orig(cortexInds, 1:n) * betaCoeffs_ortho{n};
   corrCoeffs_ortho(n,:) = diag(corr(empiricalData(cortexInds, :), squeeze(recon_ortho(:, n, :))));
end

%% Add medial wall back into reconstructed data

recon_full = nan(nVerts_orig, nModes, nImages);
recon_full(cortexInds, :, :) = recon_orig;
recon_orig = recon_full;

recon_full = nan(nVerts_orig, nModes, nImages);
recon_full(cortexInds, :, :) = recon_ortho;
recon_ortho = recon_full;

%% Plot recon accuracy using stacked bar chart
modeReconPoints = [5, 10, 20, 50, 100];
plotTargetMaps = 0;

for ii = 1:2
    reconCombinedStacked = nan(nImages, 2, length(modeReconPoints));

    % Calculate mode reconstruction accuracies for each recon point
    for jj = 1:length(modeReconPoints)
        if jj == 1
            prevAcc_orig = zeros([1, nImages]);
            prevAcc_ortho = zeros([1, nImages]);
        else
            prevAcc_orig = corrCoeffs_orig(modeReconPoints(jj - 1), :);
            prevAcc_ortho = corrCoeffs_ortho(modeReconPoints(jj - 1), :);
        end
        % Subtract corr accuracy at previous modeReconPoint from the corr accuracy at the current
        % modeReconPoint
        reconCombinedStacked(:, 1, jj) = corrCoeffs_orig(modeReconPoints(jj), :) - prevAcc_orig;
        reconCombinedStacked(:, 2, jj) = corrCoeffs_ortho(modeReconPoints(jj), :) - prevAcc_ortho; 
    end

    figure('Position', [200, 200, 1600, 800], 'visible', 'on');
    nRows = 6;
    tl1 = tiledlayout(nRows, 1, 'TileSpacing', 'none');

    if plotTargetMaps
        % Plot the target maps
        tl2 = tiledlayout(tl1, 1, nImages, 'TileSpacing', 'tight');
        tl2.Layout.Tile = 1;
        for jj = 1:nImages
            % Calculate clims based on min and max values of cortexInds. Some maps have a bunch of zero
            % values within the cortex indices. Omit these from the clims calculations.
            cortexData = empiricalData(cortexInds, jj);
            isZero = (abs(cortexData) < 1e-4);
            nonZeroData = cortexData(~isZero);
            clims = [min(nonZeroData), max(nonZeroData)];
            % Hard code max colour limit for specific maps for better visualisation
            if empiricalLabels{jj} == "T1w/T2w"
                clims(2) = 1.7;
            elseif empiricalLabels{jj} == "CBF"
                clims(1) = 4000;
            end
    
            [~, ~, tl3, tl4] = plotBrain('lh', {surface, medialMask, empiricalData(:, jj)}, 'parent', tl2, ...
                'colormap', viridis, 'view', {'ll', 'lm'}, 'tiledLayoutOptions', {1, 1, 'TileSpacing', 'none'}, ...
                'tiledLayout2Options', {1, 2, 'TileSpacing', 'none'}, 'clim', clims, 'groupBy', 'data'); 
            tl3.Layout.Tile = jj;
        end
    end

    % Plot the recon accuracy bars
    ax = nexttile(tl1, [nRows-1, 1]);
    h = plotBarStackGroups(reconCombinedStacked, empiricalLabels);
    
    % TODO: use make_cmap('red', 5)
    blue_map = cat(1, [239, 243, 255]/255, [189,215,231]/255, [107,174,214]/255, [49,130,189]/255, [8,81,156]/255);
    red_map = cat(1, [254,229,217]/255, [252,174,145]/255, [251,106,74]/255, [222,45,38]/255, [165,15,21]/255);
    
    % Set colours of bar segments to blue and red shades
    set(h(:,:), 'FaceColor', 'Flat')
    for jj = 1:size(blue_map, 1)
        h(1, jj).CData = blue_map(jj, :);
        h(2, jj).CData = red_map(jj, :);
    end
    
    xLabs = get(ax, 'XTickLabel');
    set(ax, 'XTickLabel', xLabs, 'FontSize', 12, 'TickLabelInterpreter', 'none');
    xlabel("Reconstructed map", "FontSize", 15)
    ylabel("Reconstruction accuracy", "FontSize", 15) 
    lgd = legend('5 modes', '10 modes', '20 modes', '50 modes', '100 modes', ...
        '5 modes', '10 modes', '20 modes', '50 modes', '100 modes');
    lgd.NumColumns = 2; lgd.Location = "eastoutside"; lgd.FontSize = 12;
    title(lgd, "Original    Normalized", 'FontSize', 15)
    
end

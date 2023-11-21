%% Code to understand what adding heterogeneity into eigenmodes actually does
disp("running....")

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

heteroLabel = "myelinmap"; % only plot one hetero map per figure
scale = ["cmean"];
alphaVals = 0.1:0.1:0.2;
% alphaVals = ["0.1", "0.5", "1.0", "1.5", "2.0", "2.5"];
dset = "hcp-tfMRI";

% Set text for output fileneames
heteroParams = {heteroLabel, scale};
text = cell(1, numel(heteroParams));
for ii = 1:numel(heteroParams)
    if size(heteroParams{ii}, 2) > 1
        text{ii} = strjoin(heteroParams{ii}, '-');
    else
        text{ii} = heteroParams{ii};
    end
end
heteroText = text{1};
scaleText = text{2};

% TODO: set list of basis set filenames
nBasisSets = length(alphaVals);
basisSetLabels = alphaVals;

disp("Loading modes and empirical data...")

% Get cortex indices
medialMask = dlmread(sprintf('%s/atlas-yeo_space-%s_den-%s_hemi-%s_medialMask.txt', surfDir, space, ...
    den, hemi));
cortexInds = find(medialMask);

% Load geometric eigenmodes and eigenvalues
geomDesc = 'hetero-%s_atlas-%s_space-%s_den-%s_surf-%s_hemi-%s_n-%i_maskMed-True';
geomModes = dlmread(fullfile(emodeDir, sprintf(geomDesc, "None", atlas, space, den, surf, hemi, nModes) ...
    + "_emodes.txt"));
geomEvals = dlmread(fullfile(emodeDir, sprintf(geomDesc, "None", atlas, space, den, surf, hemi, nModes) ...
    + "_evals.txt"));

% Load heterogeneous eigenmodes and eigenvalues
heteroDesc = 'hetero-%s_atlas-%s_space-%s_den-%s_surf-%s_hemi-%s_n-%i_scale-%s_alpha-%.1f_maskMed-True';
heteroModes = zeros([size(geomModes), nBasisSets]);
heteroEvals = zeros(nBasisSets, nModes);
for ii=1:nBasisSets  
    heteroModes(:, :, ii) = dlmread(fullfile(emodeDir, sprintf(heteroDesc, heteroLabel, atlas, ...
        space, den, surf, hemi, nModes, scale, alphaVals(ii)) + "_emodes.txt")); 
    heteroEvals(ii, :) = dlmread(fullfile(emodeDir, sprintf(heteroDesc, heteroLabel, atlas, space, ...
        den, surf, hemi, nModes, scale, alphaVals(ii)) + "_evals.txt")); 
end

% Load empirical data to reconstruct
if dset == "nm-images" || dset == "nm-subset"
    % Load neuromaps empiricalData
    empiricalData = load_images('/fs03/kg98/vbarnes/HBO_data/neuromaps-data/resampled/fsLR/dset-nm_desc-noPET_space-fsLR_den-32k_resampled.csv');
    
    if dset == "nm-subset"
        all_empiricalData = empiricalData;
        % Get subset of neuromaps data
        subset = {'megalpha', 'fcgradient01', 'myelinmap', 'evoexp_xu2020',...
            'scalinghcp'};
        empiricalData = struct();
        for ii = 1:numel(subset)
            label = subset{ii};
            empiricalData.(label) = all_empiricalData.(label);
        end
    end
elseif dset == "hcp-tfMRI"
    pang2023Dir = '/fs03/kg98/vbarnes/repos/BrainEigenmodes';
    % Load hcp task fMRI data
    empiricalData = load(sprintf('%s/data/empirical/S255_tfMRI_ALLTASKS_raw_lh.mat', pang2023Dir));
    empiricalData = empiricalData.zstat;
    % Only load first subject for each task contrast
    contrasts = fieldnames(empiricalData);
    for ii=1:numel(contrasts)
        empiricalData.(contrasts{ii}) = empiricalData.(contrasts{ii})(:, 1);
    end
end
empiricalLabels = fieldnames(empiricalData);
nImages = numel(empiricalLabels);

%% Calculate recon beta coefficients using 1 to num_modes eigenmodes
tic
disp("Computing beta coefficients...")
geomReconBeta = zeros(nModes, nModes, nImages);
heteroReconBeta = zeros(nModes, nModes, nImages);

for ii = 1:length(alphaVals)
    for jj = 1:nImages
        image = empiricalData.(empiricalLabels{jj});
        disp(empiricalLabels{jj})
        
        % If an image contains NaNs then set method to 'regression' to avoid errors
        if any(isnan(image(cortexInds)))
            method = 'regression';
            disp('Found NaNs... Using regression')
        else
            method = 'matrix';
        end
        
        for mode = 1:nModes
            % Calculate beta spectra of empirical data using geometric modes
            geomReconBeta(1:mode, mode, jj) = calc_eigendecomposition(image(cortexInds), ...
                geomModes(cortexInds, 1:mode), method);
            % Calculate beta spectra of empirical data using hetero modes
            heteroReconBeta(1:mode, mode, jj) = calc_eigendecomposition(image(cortexInds), ...
                heteroModes(cortexInds, 1:mode, ii), method);
        end
    end
    fprintf("Finished computing beta coefficients: %.2f mins\n", toc/60)

    %% Calculate reconstruction accuracy using 1 to num_modes eigenmodes
    tic
    disp("Computing reconstruction accuracy...")
    geomReconCorrs = zeros(nImages, nModes);   
    heteroReconCorrs = zeros(nImages, nModes);
    for jj = 1:nImages
        image = empiricalData.(empiricalLabels{jj});
        for mode = 1:nModes
            reconTemp = geomModes(cortexInds, 1:mode)*geomReconBeta(1:mode, mode, jj);
            geomReconCorrs(jj, mode) = corr(image(cortexInds), reconTemp, "rows", "complete");

            reconTemp = heteroModes(cortexInds, 1:mode)*heteroReconBeta(1:mode, mode, jj);
            heteroReconCorrs(jj, mode) = corr(image(cortexInds), reconTemp, "rows", "complete");
        end
    end
    fprintf("Finished computing reconsruction corrs: %.2f mins\n", toc/60)

    % Change labels to plotting labels for clearer figures
    if dset == "nm_images" || dset == "nm_subset"
        config_file = fileread('/fs03/kg98/vbarnes/human-brain-observatory/scripts/configurations/configHBO.json');
        configHBO = jsondecode(config_file);
        plotting_labels = configHBO.neuromaps.plotting_labels;
        for jj = 1:length(empiricalLabels)
            original_label = empiricalLabels{jj};
            empiricalLabels{jj} = plotting_labels.(original_label);
        end
    end

    fprintf("Finished computing reconstruction difference: %.2f mins\n", toc/60)

    %% Compare reconstruction accuracy and plot
    fig = figure('Position', [200 200 1000 600]);

    recon_corr_combined = cat(3, heteroReconCorrs - geomReconCorrs);

    % Set clims
    clims_combined = zeros(2,2);
    for kk=1:1
        recon_corr = recon_corr_combined(:, :, kk);
        clim_max = max(recon_corr(:,2:end),[],'all');
        clim_min = min(recon_corr(:,2:end),[],'all');
        clims_combined(kk,:) = [clim_min, clim_max];
    end

    tl = tiledlayout(1, 1);
    for kk=1:1   
        data_to_plot = recon_corr_combined(:,:,kk);
        %     clim = clims_combined(kk,:);
        % Make absolute colorbar for all two eigenmodes
        clim = [min(clims_combined(:,1)), max(clims_combined(:,2))]; 
        % Make colorbar symmetric according to the maximum absolute value
        clim = max(abs(min(clims_combined(:,1))), abs(max(clims_combined(:,2))))*[-1 1]; 
        
        nexttile();
        imagesc(data_to_plot)
        for jj=1:nImages-1
            yline(jj+0.5, 'k-');
        end

        caxis(clim)
        colormap(bluewhitered)
        % cbar = colorbar;
        set(gca, 'fontsize', 10, 'ticklength', [0.02, 0.02], 'xlim', [1, nModes], 'ytick', 1:nImages,...
            'yticklabel', empiricalLabels, 'ticklabelinterpreter', 'none')
        if kk~=1
            set(gca, 'yticklabel', {})
        end
        xlabel(tl, 'number of modes')
        title({'Heterogeneous modes'; sprintf('(%s | alpha: %.1f)', heteroLabel, alphaVals)}, 'fontweight', 'normal', 'fontsize', 18)
    end

    cbar = colorbar('Location', 'eastoutside');
    ylabel(cbar, 'reconstruction accuracy difference', 'fontsize', 16)

    % Save image
    savecf(sprintf("%s/results/evaluation/hetero-%s_surf-%s_scale-%s_alpha-%.1f_empiricalData-%s_reconAccuracyDiff", ...
        projDir, heteroText, surf, scaleText, alphaVals(ii), dset), ".png", 200)
end

% %% Plot reconstruction acuracy paths
% figure; 

% % Hetero modes recon accuracy
% subplot(1, 2, 1); 
% hold on; 
% for i=1:n_images
%     plot(recon_corr_hetero(i, 2:n_modes)); 
% end
% hold off; 
% xlabel("number of modes"); title("Heterogeneous"); pbaspect([1 1 1]);
% ylabel("reconstruction accuracy");
% ylim([0, 1])
% legend(labels, "location", "southeast", 'Interpreter', 'none')

% % Geometric modes recon accuracy
% subplot(1, 2, 2); 
% hold on; 
% for i=1:n_images
%     plot(recon_corr_geom(i, 2:n_modes)); 
% end
% hold off; 
% xlabel("number of modes"); title("Geometric"); pbaspect([1 1 1]);
% ylabel("reconstruction accuracy");
% ylim([0, 1])
% legend(labels, "location", "southeast", 'Interpreter', 'none')

%% Add required functions to path

libraries = [
    "NSB_utils_matlab", ...
    "gifti", ...
    "BrainSurfaceAnimation", ...
    "nitoolsmat", ...
    "Colormaps" ...
    ];

% Add folder WITHOUT subfolders to path
addpath(reposDir + filesep + "BrainEigenmodes/functions_matlab")

% Add foler WITH subfolders to path (using genpath)
for ii = libraries
    addpath(genpath(reposDir + filesep + ii));
end

clear ii libraries projectDirLibraries

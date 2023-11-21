%% Add required functions to path

libraries = [
    "repos/BMH_utils_matlab", ...
    "repos/gifti", ...
    "repos/BrainSurfaceAnimation", ...
    "repos/nitoolsmat" ...
    ];

% Add folder WITHOUT subfolders to path
addpath(projectDirLibraries + filesep + "repos/BrainEigenmodes/functions_matlab")

% Add foler WITH subfolders to path (using genpath)
for ii = libraries
    addpath(genpath(projectDirLibraries+filesep+ii));
end

clear ii libraries projectDirLibraries

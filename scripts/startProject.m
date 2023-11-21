%% Add required functions to path
projectDirLibraries = "/fs03/kg98/vbarnes";

libraries = [
    "repos/BMH_utils_matlab", ...
    "repos/gifti", ...
    "repos/BrainSurfaceAnimation", ...
    "repos/nitoolsmat" ...
    ];

% Add folder WITHOUT subfolders
addpath(projectDirLibraries + filesep + "repos/BrainEigenmodes/functions_matlab")

% Add foler WITH subfolders (using genpath)
for ii = libraries
    addpath(genpath(projectDirLibraries+filesep+ii));
end

clear ii libraries projectDirLibraries

function phfcd = calc_phase_fcd(time_series, TR)
% calc_phase_fcd.m
%
% Calculate phase fcd of a time series.
%
% Inputs: time_series : time series [T x N]
%                       T = number of time points
%                       N = number of regions
%                            
%         TR          : repetition time (or interval between time points) (float)
%
% Output: phfcd       : phase fcd values (1D vector)
%
% Original: Kevin Aquino, Monash University, 2021
% Cleaned new version: James Pang, Monash University, 2024

%% Define parameters

k = 2;                            % 2nd order butterworth filter
fnq = 1/(2*TR);
flp = .04;                        % low-pass frequency of filter
fhi = .07;                        % high-pass frequency of the filter
Wn = [flp/fnq fhi/fnq];           % butterworth bandpass non-dimensional frequency
[bfilt2,afilt2] = butter(k,Wn);   % construct the filter
[T,N] = size(time_series);        % T = number of time points, N = number of regions

%% calculate synchrony between regions at every single time point

BOLD = time_series';
phase_BOLD = zeros(N, T);
BOLD_filtered = zeros(N, T);

% calculate phase for each region
for region = 1:N
    BOLD(region,:) = detrend(BOLD(region,:), 'constant');
    BOLD_filtered(region,:) = filtfilt(bfilt2, afilt2, BOLD(region,:));
    phase_BOLD(region,:) = angle(hilbert(BOLD_filtered(region,:)));
end

% calculate synchrony
tril_ind = find(tril(ones(N), -1));     % lower triangle index of NxN matrix
time_vec_trunc = 10:T-10;               % remove first 9 and last 9 time points (cutoff arbitrarily chosen)
num_time_points = length(time_vec_trunc);
synchrony_vec = zeros(length(time_vec_trunc), length(tril_ind));
for t_ind = 1:num_time_points
    t = time_vec_trunc(t_ind);
    
%     %%% slower version
%     synchrony_mat = zeros(N);
%     for ii=1:N
%         for jj=1:ii
%             synchrony_mat(ii,jj) = cos(phase_BOLD(ii,t)-phase_BOLD(jj,t));
%         end
%     end    
    
    %%% faster version
    phase_BOLD_repmat = repmat(phase_BOLD(:,t), 1, N);
    synchrony_mat = cos(phase_BOLD_repmat - phase_BOLD_repmat');
    
    synchrony_vec(t_ind,:) = synchrony_mat(tril_ind);
end
    
%% calculate phase from synchrony at each time point 
% mean of up to the 2nd-order forward neighbor, hence we need to remove two time
% points at the end

% %%% slower version
% phfcd = [];
% counter = 1;
% for t_ind_1 = 1:num_time_points-2
%     p1 = mean(synchrony_vec(t_ind_1:t_ind_1+2,:));
%     
%     for t_ind_2 = tt_ind_1+1:num_time_points-2
%         p2 = mean(synchrony_vec(t_ind_2:t_ind_2+2,:));
%         
%         phfcd(counter) = dot(p1,p2)/norm(p1)/norm(p2);
%         
%         counter = counter+1;
%     end
% end

%%% faster version
p_mat = zeros(num_time_points-2, size(synchrony_vec,2));

% pre-calculate phase vectors
for t_ind = 1:num_time_points-2
    p_mat(t_ind,:) = mean(synchrony_vec(t_ind:t_ind+2,:));
    p_mat(t_ind,:) = p_mat(t_ind,:)/norm(p_mat(t_ind,:));
end

% calculate phase for every time pair
phfcd_mat = p_mat*p_mat';

triu_ind = find(triu(ones(size(phfcd_mat)), 1));     % upper triangle inde
phfcd = phfcd_mat(triu_ind);


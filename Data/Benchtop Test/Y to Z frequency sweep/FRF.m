clc
clear all
close all

%% Parameters
fs = 400;        % Sampling frequency [Hz]
dt = 1/fs;       % Time step [s]

%% Import data
D1 = dlmread('C:\Users\joud\Dropbox\Satme_2024_IEEE_Edge_Processing_Compensator\Data\Benchtop Test\Y to Z frequency sweep\Ref1.csv',',',1);
D2 = dlmread('C:\Users\joud\Dropbox\Satme_2024_IEEE_Edge_Processing_Compensator\Data\Benchtop Test\Y to Z frequency sweep\Pkg1.csv',',',1);

%% Extract acceleration data
input_signal  = D1(:,1);
output_signal = D2(:,1);

%% Remove the mean (DC offset)
input_signal  = input_signal  - mean(input_signal);
output_signal = output_signal - mean(output_signal);

%% Match signal lengths
N = min(length(input_signal), length(output_signal));  % use the shorter length
input_signal  = input_signal(1:N);
output_signal = output_signal(1:N);

%% Build time vector
t = (0:N-1)*dt;   % time vector

%% --- TIME DOMAIN PLOT ---
figure;
plot(t, input_signal, 'b', 'LineWidth', 1.2); hold on;
plot(t, output_signal, 'r', 'LineWidth', 1.2);
xlabel('Time [s]');
ylabel('Acceleration [m/s^2]');
title('Input vs Output Acceleration (Mean Removed)');
legend('Input','Output');
grid on;

%% --- FREQUENCY DOMAIN (FFT) ---
% Compute FFT
Y_in  = fft(input_signal);
Y_out = fft(output_signal);

% Compute two-sided spectrum, then single-sided magnitude spectrum
P2_in  = abs(Y_in/N);
P2_out = abs(Y_out/N);

P1_in  = P2_in(1:floor(N/2)+1);    
P1_out = P2_out(1:floor(N/2)+1);

P1_in(2:end-1)  = 2*P1_in(2:end-1);    
P1_out(2:end-1) = 2*P1_out(2:end-1);

% Frequency vector
f = fs*(0:floor(N/2))/N;

%% Limit to Hz
max_freq = 20;
idx_limit = f <= max_freq;

f_plot    = f(idx_limit);
P1_in_plot  = P1_in(idx_limit);
P1_out_plot = P1_out(idx_limit);

%% --- FREQUENCY DOMAIN PLOT (up to 23 Hz) ---
figure;
plot(f_plot, P1_in_plot, 'b', 'LineWidth', 1.2); hold on;
plot(f_plot, P1_out_plot, 'r', 'LineWidth', 1.2);
xlim([0 20]);
xlabel('Frequency [Hz]');
ylabel('Magnitude');
title('Single-Sided Amplitude Spectrum (up to 5 Hz)');
legend('Input','Output');
grid on;

%% --- FREQUENCY RESPONSE FUNCTION (FRF) ---
% FRF = Output / Input in frequency domain (complex)
FRF1 = Y_out ./ Y_in;       % complex FRF
FRF1 = FRF1(1:floor(N/2)+1); % single-sided
FRF_plot = FRF1(idx_limit); % only up to 23 Hz

% Magnitude and Phase of FRF
FRF_mag   = abs(FRF_plot);
FRF_phase = angle(FRF_plot) * (180/pi);  % convert to degrees

%% --- PLOT FRF MAGNITUDE & PHASE (Smoothed) ---
% Apply 100-point moving average to FRF magnitude
FRF_mag_smooth = movmean(FRF_mag, 50);

figure;
plot(f_plot, FRF_mag_smooth, 'k', 'LineWidth', 1.2);
ylim([0 10]);
xlim([0 20]);
xlabel('Frequency [Hz]');
ylabel('|FRF| (Smoothed)');
title('Frequency Response Function - Magnitude (100-point Moving Average)');
grid on;



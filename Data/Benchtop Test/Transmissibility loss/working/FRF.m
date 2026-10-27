clc
clear all
close all

%% Parameters
fs = 400;        % Sampling frequency [Hz]
dt = 1/fs;       % Time step [s]

%% Import data

% Relative paths -_-
D1 = dlmread('./measured.csv');
D2 = dlmread('./uncompressed.csv');
D3 = dlmread('./reduced_15.csv');
D4 = dlmread('./true.csv');

%% Extract acceleration data
Measured     = D1(:,1);
Uncompressed = D2(:,1);
Reduced      = D3(:,1);
Ref          = D4(:,1);

%% Remove DC offset
Measured     = Measured     - mean(Measured);
Uncompressed = Uncompressed - mean(Uncompressed);
Reduced      = Reduced      - mean(Reduced);
Ref          = Ref          - mean(Ref);

%% Match signal lengths
N = min([length(Measured), length(Uncompressed), length(Reduced), length(Ref)]);
Measured     = Measured(1:N);
Uncompressed = Uncompressed(1:N);
Reduced      = Reduced(1:N);
Ref          = Ref(1:N);

%% Time vector
t = (0:N-1)*dt;

%% --- TIME DOMAIN PLOT ---
figure;
plot(t, Measured, 'r', 'LineWidth', 1.2); hold on;
plot(t, Uncompressed, 'b', 'LineWidth', 1.2);
plot(t, Reduced, 'g', 'LineWidth', 1.2);
plot(t, Ref, 'k', 'LineWidth', 1.2);
xlabel('Time [s]');
ylabel('Acceleration [m/s^2]');
title('Time Domain Signals');
legend('Measured','Uncompressed','Reduced','Reference');
grid on;

%% --- FFT of All Signals ---
Y_measured     = fft(Measured);
Y_uncompressed = fft(Uncompressed);
Y_reduced      = fft(Reduced);
Y_ref          = fft(Ref);

% Magnitude spectra
P2_measured     = abs(Y_measured/N);
P2_uncompressed = abs(Y_uncompressed/N);
P2_reduced      = abs(Y_reduced/N);
P2_ref          = abs(Y_ref/N);

P1_measured     = P2_measured(1:floor(N/2)+1);
P1_uncompressed = P2_uncompressed(1:floor(N/2)+1);
P1_reduced      = P2_reduced(1:floor(N/2)+1);
P1_ref          = P2_ref(1:floor(N/2)+1);

% Double non-DC and Nyquist components
P1_measured(2:end-1)     = 2 * P1_measured(2:end-1);
P1_uncompressed(2:end-1) = 2 * P1_uncompressed(2:end-1);
P1_reduced(2:end-1)      = 2 * P1_reduced(2:end-1);
P1_ref(2:end-1)          = 2 * P1_ref(2:end-1);

% Frequency vector
f = fs * (0:floor(N/2)) / N;

% Limit to max freq
max_freq = 20;
idx_limit = f <= max_freq;
f_plot = f(idx_limit);

%% --- FREQUENCY DOMAIN PLOT ---
figure;
plot(f_plot, P1_measured(idx_limit), 'r', 'LineWidth', 1.2); hold on;
plot(f_plot, P1_uncompressed(idx_limit), 'b', 'LineWidth', 1.2);
plot(f_plot, P1_reduced(idx_limit), 'g', 'LineWidth', 1.2);
plot(f_plot, P1_ref(idx_limit), 'k', 'LineWidth', 1.2);
xlabel('Frequency [Hz]');
ylabel('Magnitude');
title('Single-Sided Amplitude Spectrum');
legend('Measured','Uncompressed','Reduced','Reference');
xlim([0 max_freq]);
grid on;

%% --- FREQUENCY RESPONSE FUNCTIONS (FRFs) ---
FRF_measured     = Y_measured     ./ Y_ref;
FRF_uncompressed = Y_uncompressed ./ Y_ref;
FRF_reduced      = Y_reduced      ./ Y_ref;

FRF_measured     = FRF_measured(1:floor(N/2)+1);
FRF_uncompressed = FRF_uncompressed(1:floor(N/2)+1);
FRF_reduced      = FRF_reduced(1:floor(N/2)+1);

FRF_measured_mag_smooth     = movmean(abs(FRF_measured(idx_limit)),     100);
FRF_uncompressed_mag_smooth = movmean(abs(FRF_uncompressed(idx_limit)), 100);
FRF_reduced_mag_smooth      = movmean(abs(FRF_reduced(idx_limit)),      100);

%% --- FRF MAGNITUDE PLOT ---
figure;
plot(f_plot, FRF_measured_mag_smooth, 'r', 'LineWidth', 1.2); hold on;
plot(f_plot, FRF_uncompressed_mag_smooth, 'b', 'LineWidth', 1.2);
plot(f_plot, FRF_reduced_mag_smooth, 'g', 'LineWidth', 1.2);
xlabel('Frequency [Hz]');
ylabel('|FRF| (Smoothed)');
title('Smoothed FRF Magnitude Compared to Reference');
legend('Raw/Ref','Compensator/Ref','Reduced Compensator/Ref');
xlim([0 max_freq]);
ylim([0 10]);
grid on;

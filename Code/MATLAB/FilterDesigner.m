% ============================================
% Import Time-Domain Acceleration Signal
% Apply High-Pass Filters
% and Plot Time + Frequency Domains
% ============================================

clc;
clear;
close all;

% File path
filePath = 'C:\Users\joud\Dropbox\Satme_2026_IEEE_Edge_Processing_Compensator\Data\Benchtop Test\Final Results\Time_Domain V5.csv';
filePath = 'C:\Users\joud\Dropbox\Satme_2026_IEEE_Edge_Processing_Compensator\Data\Benchtop Test\Test Sweep\0-20Pkg.CSV';
filePath = 'C:\Users\joud\Dropbox\Satme_2026_IEEE_Edge_Processing_Compensator\Data\Benchtop Test\Train Sweep\0-20Pkg.CSV';

% Read CSV data while skipping first line
data = readmatrix(filePath, 'NumHeaderLines', 1);

% Column assignments
time = data(:,1);          % First column = time
acceleration = data(:,2);  % Second column = acceleration

% --------------------------------------------
% Remove Mean from Signal
% --------------------------------------------

acceleration = acceleration - mean(acceleration);

% --------------------------------------------
% Frequency Domain Calculation (FFT)
% --------------------------------------------

% Sampling parameters
dt = mean(diff(time));     % Sampling interval
Fs = 1/dt;                 % Sampling frequency
N = length(acceleration);  % Number of samples

% --------------------------------------------
% Design High-Pass Butterworth Filters
% --------------------------------------------

fc = 10;                   % Cutoff frequency in Hz
Wn = fc/(Fs/2);            % Normalized cutoff frequency

[b3,a3] = butter(2, Wn, 'high');
[b4,a4] = butter(5, Wn, 'high');
[b5,a5] = butter(8, Wn, 'high');

% --------------------------------------------
% Apply Filters
% --------------------------------------------

accel_hp3 = filtfilt(b3, a3, acceleration);
accel_hp4 = filtfilt(b4, a4, acceleration);
accel_hp5 = filtfilt(b5, a5, acceleration);

% --------------------------------------------
% FFT of Reference and Filtered Signals
% --------------------------------------------

% Reference
Y_ref = fft(acceleration);
P2_ref = abs(Y_ref/N);
P1_ref = P2_ref(1:N/2+1);
P1_ref(2:end-1) = 2*P1_ref(2:end-1);

% 2nd Order
Y_hp3 = fft(accel_hp3);
P2_hp3 = abs(Y_hp3/N);
P1_hp3 = P2_hp3(1:N/2+1);
P1_hp3(2:end-1) = 2*P1_hp3(2:end-1);

% 5th Order
Y_hp4 = fft(accel_hp4);
P2_hp4 = abs(Y_hp4/N);
P1_hp4 = P2_hp4(1:N/2+1);
P1_hp4(2:end-1) = 2*P1_hp4(2:end-1);

% 8th Order
Y_hp5 = fft(accel_hp5);
P2_hp5 = abs(Y_hp5/N);
P1_hp5 = P2_hp5(1:N/2+1);
P1_hp5(2:end-1) = 2*P1_hp5(2:end-1);

% Frequency vector
f = Fs*(0:(N/2))/N;

% --------------------------------------------
% Filter Magnitude Responses
% --------------------------------------------

[H3,F3] = freqz(b3, a3, 4096, Fs);
[H4,F4] = freqz(b4, a4, 4096, Fs);
[H5,F5] = freqz(b5, a5, 4096, Fs);

% --------------------------------------------
% Plotting
% --------------------------------------------

figure;

% --------------------------------------------
% Time Domain Plot
% --------------------------------------------

subplot(3,1,1);

plot(time, acceleration, 'LineWidth', 1.2);
hold on;
%plot(time, accel_hp3, 'LineWidth', 1.2);
%plot(time, accel_hp4, 'LineWidth', 1.2);
plot(time, accel_hp5, 'LineWidth', 1.2);
hold off;

xlabel('time(s)');
ylabel('acceleration');
title('Time-Domain Signal');
xlim([0,182]);

legend('Reference', '5th Order HPF',...
       'Location', 'northwest');

grid on;

% --------------------------------------------
% Filter Magnitude Response Plot
% --------------------------------------------

subplot(3,1,2);

% Unfiltered reference magnitude
plot([0 max(F5)], [1 1], '--', 'LineWidth', 1.2);
hold on;
%plot(F3, abs(H3), 'LineWidth', 1.2);
hold on;
%plot(F4, abs(H4), 'LineWidth', 1.2);
hold on;
plot(F5, abs(H5), 'LineWidth', 1.2);



hold off;

xlabel('frequency (Hz)');
ylabel('magnitude');
title('Filter Magnitude Responses');

xlim([0.1,21]);
ylim([-0.1,1.1]);

legend('Unfiltered Reference','5th Order HPF', ...
    'Location', 'SouthEast');

grid on;

% --------------------------------------------
% Frequency Domain Plot
% --------------------------------------------

subplot(3,1,3);

plot(f, P1_ref, 'LineWidth', 1.2);
hold on;
%plot(f, P1_hp3, 'LineWidth', 1.2);
%plot(f, P1_hp4, 'LineWidth', 1.2);
plot(f, P1_hp5, 'LineWidth', 1.2);
hold off;

xlabel('frequency (Hz)');
ylabel('amplitude');
title('Frequency-Domain Signal');

xlim([0.1,21]);

legend('Reference', '5th Order HPF', ...
       'Location', 'northwest');

grid on;

disp('Signal imported, mean removed, filtered, and processed successfully.');

% --------------------------------------------
% Figure Formatting
% --------------------------------------------
set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');

% --------------------------------------------
% Figure Export Settings
% --------------------------------------------

set(gcf, 'Units', 'inches');
set(gcf, 'Position', [1 1 6.5 7]);

% Ensure saved size matches figure size
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [6.5 7]);
set(gcf, 'PaperPosition', [0 0 6.5 7]);

% Export as 300 DPI PNG
print(gcf, 'Filtered_Signal_Comparison.png', '-dpng', '-r300');

% Optional: export as vector PDF
print(gcf, 'Filtered_Signal_Comparison.pdf', '-dpdf', '-bestfit');



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ============================================
% NEW FIGURE
% 5th Order High-Pass Filters
% Varying Cutoff Frequencies
% ============================================

% --------------------------------------------
% Design 5th Order High-Pass Filters
% --------------------------------------------

fc1 = 5;
fc2 = 10;
fc3 = 15;

Wn1 = fc1/(Fs/2);
Wn2 = fc2/(Fs/2);
Wn3 = fc3/(Fs/2);

[b1,a1] = butter(5, Wn1, 'high');
[b2,a2] = butter(5, Wn2, 'high');
[b3,a3] = butter(5, Wn3, 'high');

% --------------------------------------------
% Apply Filters
% --------------------------------------------

accel_hp_5Hz  = filtfilt(b1, a1, acceleration);
accel_hp_10Hz = filtfilt(b2, a2, acceleration);
accel_hp_15Hz = filtfilt(b3, a3, acceleration);

% --------------------------------------------
% FFT Calculations
% --------------------------------------------

% 5 Hz
Y_5Hz = fft(accel_hp_5Hz);
P2_5Hz = abs(Y_5Hz/N);
P1_5Hz = P2_5Hz(1:N/2+1);
P1_5Hz(2:end-1) = 2*P1_5Hz(2:end-1);

% 10 Hz
Y_10Hz = fft(accel_hp_10Hz);
P2_10Hz = abs(Y_10Hz/N);
P1_10Hz = P2_10Hz(1:N/2+1);
P1_10Hz(2:end-1) = 2*P1_10Hz(2:end-1);

% 15 Hz
Y_15Hz = fft(accel_hp_15Hz);
P2_15Hz = abs(Y_15Hz/N);
P1_15Hz = P2_15Hz(1:N/2+1);
P1_15Hz(2:end-1) = 2*P1_15Hz(2:end-1);

% --------------------------------------------
% Filter Magnitude Responses
% --------------------------------------------

[H1,F1] = freqz(b1, a1, 4096, Fs);
[H2,F2] = freqz(b2, a2, 4096, Fs);
[H3,F3] = freqz(b3, a3, 4096, Fs);

% --------------------------------------------
% Plotting
% --------------------------------------------

figure;

% --------------------------------------------
% Time Domain Plot
% --------------------------------------------

subplot(3,1,1);

plot(time, acceleration, 'LineWidth', 1.2);
hold on;
plot(time, accel_hp_5Hz,  'LineWidth', 1.2);
plot(time, accel_hp_10Hz, 'LineWidth', 1.2);
plot(time, accel_hp_15Hz, 'LineWidth', 1.2);
hold off;

xlabel('time(s)');
ylabel('acceleration');
title('Time-Domain Signal');

xlim([0,182]);

legend('Reference', ...
       '5 Hz Cutoff', ...
       '10 Hz Cutoff', ...
       '15 Hz Cutoff', ...
       'Location', 'northwest');

grid on;

% --------------------------------------------
% Filter Magnitude Response Plot
% --------------------------------------------

subplot(3,1,2);

plot([0 max(F3)], [1 1], '--', 'LineWidth', 1.2);
hold on;

plot(F1, abs(H1), 'LineWidth', 1.2);
plot(F2, abs(H2), 'LineWidth', 1.2);
plot(F3, abs(H3), 'LineWidth', 1.2);

hold off;

xlabel('frequency (Hz)');
ylabel('magnitude');
title('Filter Magnitude Responses');

xlim([0.1,21]);

legend('Unfiltered Reference', ...
       '5 Hz Cutoff', ...
       '10 Hz Cutoff', ...
       '15 Hz Cutoff', ...
       'Location', 'northwest');

grid on;

% --------------------------------------------
% Frequency Domain Plot
% --------------------------------------------

subplot(3,1,3);

plot(f, P1_ref, 'LineWidth', 1.2);
hold on;

plot(f, P1_5Hz,  'LineWidth', 1.2);
plot(f, P1_10Hz, 'LineWidth', 1.2);
plot(f, P1_15Hz, 'LineWidth', 1.2);

hold off;

xlabel('frequency (Hz)');
ylabel('amplitude');
title('Frequency-Domain Signal');

xlim([0.1,21]);

legend('Reference', ...
       '5 Hz Cutoff', ...
       '10 Hz Cutoff', ...
       '15 Hz Cutoff', ...
       'Location', 'northwest');

grid on;

% --------------------------------------------
% Figure Export Settings
% --------------------------------------------

set(gcf, 'Units', 'inches');
set(gcf, 'Position', [1 1 6.5 7]);

set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [6.5 7]);
set(gcf, 'PaperPosition', [0 0 6.5 7]);

% Export as 300 DPI PNG
print(gcf, 'Cutoff_Frequency_Comparison.png', '-dpng', '-r300');

% Optional PDF Export
print(gcf, 'Cutoff_Frequency_Comparison.pdf', '-dpdf', '-bestfit');

% ============================================
% NEW SUBSECTION
% 5th Order High-Pass Filters
% Varying Cutoff Frequencies: 2 to 20 Hz
% ============================================

% --------------------------------------------
% Define Cutoff Frequencies
% --------------------------------------------

cutoff_freqs = 2:2:20;     % 2, 4, 6, ..., 20 Hz
filter_order = 5;

num_filters = length(cutoff_freqs);

% Preallocate
accel_hp_cutoffs = zeros(N, num_filters);
P1_cutoffs = zeros(length(f), num_filters);
H_cutoffs = zeros(4096, num_filters);
F_cutoffs = zeros(4096, num_filters);

% --------------------------------------------
% Design, Apply, and Process Filters
% --------------------------------------------

for k = 1:num_filters

    fc = cutoff_freqs(k);
    Wn = fc/(Fs/2);

    [b,a] = butter(filter_order, Wn, 'high');

    % Apply filter
    accel_hp_cutoffs(:,k) = filtfilt(b, a, acceleration);

    % FFT
    Y_temp = fft(accel_hp_cutoffs(:,k));
    P2_temp = abs(Y_temp/N);
    P1_temp = P2_temp(1:N/2+1);
    P1_temp(2:end-1) = 2*P1_temp(2:end-1);

    P1_cutoffs(:,k) = P1_temp;

    % Filter magnitude response
    [H_temp,F_temp] = freqz(b, a, 4096, Fs);

    H_cutoffs(:,k) = abs(H_temp);
    F_cutoffs(:,k) = F_temp;

end

% --------------------------------------------
% Plotting
% --------------------------------------------

figure;

% --------------------------------------------
% Time Domain Plot
% --------------------------------------------

subplot(3,1,1);

plot(time, acceleration, 'k', 'LineWidth', 1.2);
hold on;

for k = 1:num_filters
    plot(time, accel_hp_cutoffs(:,k), 'LineWidth', 1.0);
end

hold off;

xlabel('time(s)');
ylabel('acceleration');
title('Time-Domain Signal');

xlim([0,182]);

legend_labels = [{'Reference'}, ...
    arrayfun(@(fc) sprintf('%d Hz Cutoff', fc), cutoff_freqs, 'UniformOutput', false)];

legend(legend_labels, 'Location', 'northwest');

grid on;

% --------------------------------------------
% Filter Magnitude Response Plot
% --------------------------------------------

subplot(3,1,2);

plot([0 max(F_cutoffs(:))], [1 1], 'k--', 'LineWidth', 1.2);
hold on;

for k = 1:num_filters
    plot(F_cutoffs(:,k), H_cutoffs(:,k), 'LineWidth', 1.0);
end

hold off;

xlabel('frequency (Hz)');
ylabel('magnitude');
title('Filter Magnitude Responses');

xlim([0.1,21]);

legend(legend_labels, 'Location', 'Southeast');

grid on;

% --------------------------------------------
% Frequency Domain Plot
% --------------------------------------------

subplot(3,1,3);

plot(f, P1_ref, 'k', 'LineWidth', 1.2);
hold on;

for k = 1:num_filters
    plot(f, P1_cutoffs(:,k), 'LineWidth', 1.0);
end

hold off;

xlabel('frequency (Hz)');
ylabel('amplitude');
title('Frequency-Domain Signal');

xlim([0.1,21]);

legend(legend_labels, 'Location', 'northwest');

grid on;

% --------------------------------------------
% Figure Export Settings
% --------------------------------------------

set(gcf, 'Units', 'inches');
set(gcf, 'Position', [1 1 6.5 10]);

set(gcf, 'PaperSize', [6.5 10]);
set(gcf, 'PaperPosition', [0 0 6.5 10]);

% Export as 300 DPI PNG
print(gcf, 'Cutoff_Frequency_Comparison_10_Filters.png', '-dpng', '-r300');

% Optional PDF Export
print(gcf, 'Cutoff_Frequency_Comparison_10_Filters.pdf', '-dpdf', '-bestfit');

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ============================================
% LSTM Network
% Input  = 10 Hz High-Pass Filtered Signal
% Output = Original Mean-Removed Signal
% Goal   = Learn to Undo Filter Effect
% ============================================

% --------------------------------------------
% Select Training Data
% --------------------------------------------

input_signal  = accel_hp_cutoffs(:,5);   % 10 Hz cutoff from cutoff_freqs = 2:2:20
target_signal = acceleration;            % Original mean-removed signal

% Normalize data
input_mu = mean(input_signal);
input_sig = std(input_signal);

target_mu = mean(target_signal);
target_sig = std(target_signal);

input_norm = (input_signal - input_mu) / input_sig;
target_norm = (target_signal - target_mu) / target_sig;

% LSTM expects data as cell arrays
XTrain = {input_norm'};
YTrain = {target_norm'};

% --------------------------------------------
% Modifiable LSTM Parameters
% --------------------------------------------
numLSTMLayers  = 1;        % Number of LSTM layers
numHiddenUnits1 = 50;        % Neurons per LSTM layer
numHiddenUnits2 = 8;        % Neurons per LSTM layer
numHiddenUnits3 = 8;        % Neurons per LSTM layer



maxEpochs = 10000;
miniBatchSize = 100;
initialLearnRate = 0.001;
learnRateDropFactor = 0.5;
learnRateDropPeriod = 100;
gradientThreshold = 1;

% --------------------------------------------
% Define LSTM Network
% --------------------------------------------

layers = [
    sequenceInputLayer(1)

    lstmLayer(numHiddenUnits1, 'OutputMode', 'sequence')
    %lstmLayer(numHiddenUnits2, 'OutputMode', 'sequence')
    %lstmLayer(numHiddenUnits3, 'OutputMode', 'sequence')

    fullyConnectedLayer(1)
    regressionLayer
];

% --------------------------------------------
% Training Options
% --------------------------------------------

options = trainingOptions('adam', ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'InitialLearnRate', initialLearnRate, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', learnRateDropFactor, ...
    'LearnRateDropPeriod', learnRateDropPeriod, ...
    'GradientThreshold', gradientThreshold, ...
    'Shuffle', 'never', ...
    'Plots', 'training-progress', ...
    'Verbose', true);

% --------------------------------------------
% Train Network
% --------------------------------------------

net = trainNetwork(XTrain, YTrain, layers, options);

% --------------------------------------------
% Predict Reconstructed Signal
% --------------------------------------------

YPred_norm = predict(net, XTrain, 'MiniBatchSize', 1);
YPred_norm = YPred_norm{1}';

% Denormalize prediction
accel_lstm_reconstructed = YPred_norm * target_sig + target_mu;

%% --------------------------------------------
% Plot LSTM Reconstruction
% --------------------------------------------

figure;

subplot(2,1,1);
plot(time, acceleration, 'LineWidth', 1.2);
hold on;
plot(time, input_signal, 'LineWidth', 1.2);
plot(time, accel_lstm_reconstructed, 'LineWidth', 1.2);
hold off;

xlabel('time(s)');
ylabel('acceleration');
title('LSTM Reconstruction: Time Domain');
xlim([0,182]);
legend('Original Reference', '10 Hz Filtered Input', 'LSTM Reconstructed Output', ...
       'Location', 'northwest');
grid on;

% FFT of LSTM reconstructed signal
Y_lstm = fft(accel_lstm_reconstructed);
P2_lstm = abs(Y_lstm/N);
P1_lstm = P2_lstm(1:N/2+1);
P1_lstm(2:end-1) = 2*P1_lstm(2:end-1);

subplot(2,1,2);
plot(f, P1_ref, 'LineWidth', 1.2);
hold on;
plot(f, P1_cutoffs(:,5), 'LineWidth', 1.2);
plot(f, P1_lstm, 'LineWidth', 1.2);
hold off;

xlabel('frequency (Hz)');
ylabel('amplitude');
title('LSTM Reconstruction: Frequency Domain');
xlim([0.1,21]);
ylim([0,0.012]);
legend('Original Reference', '10 Hz Filtered Input', 'LSTM Reconstructed Output', ...
       'Location', 'northwest');
grid on;

set(gcf, 'Units', 'inches');
set(gcf, 'Position', [1 1 6.5 7]);

set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [6.5 7]);
set(gcf, 'PaperPosition', [0 0 6.5 7]);

print(gcf, 'Time_Domain_FRF_Filter_vs_LSTM.png', '-dpng', '-r300');

%% ============================================
% FRF Comparison
% Filter vs Original
% LSTM vs Original
% ============================================

% --------------------------------------------
% FRF Calculations
% --------------------------------------------

epsilon = 1e-12;

FRF_filter = P1_cutoffs(:,5) ./ (P1_ref + epsilon);
FRF_lstm   = P1_lstm ./ (P1_ref + epsilon);

% --------------------------------------------
% Plot FRFs
% --------------------------------------------

figure;

% --------------------------------------------
% Full Frequency Range
% --------------------------------------------

subplot(2,1,1);

plot(f, abs(FRF_filter), 'LineWidth', 1.2);
hold on;
plot(f, abs(FRF_lstm), 'LineWidth', 1.2);
hold off;

xlabel('frequency (Hz)');
ylabel('FRF magnitude');
title('FRF Comparison (0-20 Hz)');

xlim([0,20]);
ylim([0,20]);
legend('10 Hz Filter / Original', ...
       'LSTM Reconstructed / Original', ...
       'Location', 'northwest');

grid on;

% --------------------------------------------
% Zoomed Frequency Range
% --------------------------------------------

subplot(2,1,2);

plot(f, abs(FRF_filter), 'LineWidth', 1.2);
hold on;
plot(f, abs(FRF_lstm), 'LineWidth', 1.2);
hold off;

xlabel('frequency (Hz)');
ylabel('FRF magnitude');
title('FRF Comparison (5-20 Hz)');

xlim([5,20]);

legend('10 Hz Filter / Original', ...
       'LSTM Reconstructed / Original', ...
       'Location', 'southeast');

grid on;

% --------------------------------------------
% Figure Export Settings
% --------------------------------------------

set(gcf, 'Units', 'inches');
set(gcf, 'Position', [1 1 6.5 7]);

set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [6.5 7]);
set(gcf, 'PaperPosition', [0 0 6.5 7]);

print(gcf, 'FRF_Filter_vs_LSTM.png', '-dpng', '-r300');


%%
% ============================================
% MLP Network
% Input  = 10 Hz High-Pass Filtered Signal
% Output = Original Mean-Removed Signal
% ============================================

input_signal  = accel_hp_cutoffs(:,5);   % 10 Hz cutoff
target_signal = acceleration;            % Original mean-removed signal

% Normalize data
input_mu  = mean(input_signal);
input_sig = std(input_signal);

target_mu  = mean(target_signal);
target_sig = std(target_signal);

input_norm  = (input_signal - input_mu) / input_sig;
target_norm = (target_signal - target_mu) / target_sig;

XTrain_MLP = input_norm;
YTrain_MLP = target_norm;

% --------------------------------------------
% Modifiable MLP Parameters
% --------------------------------------------

numNeuronsLayer1 = 8;
numNeuronsLayer2 = 8;
numNeuronsLayer3 = 8;

maxEpochs = 300;
miniBatchSize = 256;
initialLearnRate = 0.001;
learnRateDropFactor = 0.5;
learnRateDropPeriod = 100;
gradientThreshold = 1;

% --------------------------------------------
% Define MLP Network
% --------------------------------------------

layers_mlp = [
    featureInputLayer(1)

    fullyConnectedLayer(numNeuronsLayer1)
    reluLayer

    fullyConnectedLayer(numNeuronsLayer2)
    reluLayer

    fullyConnectedLayer(numNeuronsLayer3)
    reluLayer

    fullyConnectedLayer(1)
    regressionLayer
];

% --------------------------------------------
% Training Options
% --------------------------------------------

options_mlp = trainingOptions('adam', ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'InitialLearnRate', initialLearnRate, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', learnRateDropFactor, ...
    'LearnRateDropPeriod', learnRateDropPeriod, ...
    'GradientThreshold', gradientThreshold, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true);

% --------------------------------------------
% Train MLP Network
% --------------------------------------------

net_mlp = trainNetwork(XTrain_MLP, YTrain_MLP, layers_mlp, options_mlp);

% --------------------------------------------
% Predict Reconstructed Signal
% --------------------------------------------

YPred_mlp_norm = predict(net_mlp, XTrain_MLP);

accel_mlp_reconstructed = YPred_mlp_norm * target_sig + target_mu;

% --------------------------------------------
% FFT of MLP Reconstructed Signal
% --------------------------------------------

Y_mlp = fft(accel_mlp_reconstructed);
P2_mlp = abs(Y_mlp/N);
P1_mlp = P2_mlp(1:N/2+1);
P1_mlp(2:end-1) = 2*P1_mlp(2:end-1);

%% ============================================
% MLP Reconstruction Plot
% ============================================

figure;

subplot(2,1,1);

plot(time, acceleration, 'LineWidth', 1.2);
hold on;
plot(time, input_signal, 'LineWidth', 1.2);
plot(time, accel_mlp_reconstructed, 'LineWidth', 1.2);
hold off;

xlabel('time(s)');
ylabel('acceleration');
title('MLP Reconstruction: Time Domain');
xlim([0,182]);

legend('Original Reference', ...
       '10 Hz Filtered Input', ...
       'MLP Reconstructed Output', ...
       'Location', 'northwest');

grid on;

subplot(2,1,2);

plot(f, P1_ref, 'LineWidth', 1.2);
hold on;
plot(f, P1_cutoffs(:,5), 'LineWidth', 1.2);
plot(f, P1_mlp, 'LineWidth', 1.2);
hold off;

xlabel('frequency (Hz)');
ylabel('amplitude');
title('MLP Reconstruction: Frequency Domain');
xlim([0.1,21]);

legend('Original Reference', ...
       '10 Hz Filtered Input', ...
       'MLP Reconstructed Output', ...
       'Location', 'northwest');

grid on;

%% ============================================
% LSTM vs MLP vs Reference vs Filter Comparison
% ============================================

figure;

subplot(2,1,1);

plot(time, acceleration, 'LineWidth', 1.2);
hold on;
plot(time, input_signal, 'LineWidth', 1.2);
plot(time, accel_lstm_reconstructed, 'LineWidth', 1.2);
plot(time, accel_mlp_reconstructed, 'LineWidth', 1.2);
hold off;

xlabel('time(s)');
ylabel('acceleration');
title('LSTM vs MLP Reconstruction: Time Domain');
xlim([0,182]);

legend('Original Reference', ...
       '10 Hz Filtered Input', ...
       'LSTM Reconstructed Output', ...
       'MLP Reconstructed Output', ...
       'Location', 'northwest');

grid on;

subplot(2,1,2);

plot(f, P1_ref, 'LineWidth', 1.2);
hold on;
plot(f, P1_cutoffs(:,5), 'LineWidth', 1.2);
plot(f, P1_lstm, 'LineWidth', 1.2);
plot(f, P1_mlp, 'LineWidth', 1.2);
hold off;

xlabel('frequency (Hz)');
ylabel('amplitude');
title('LSTM vs MLP Reconstruction: Frequency Domain');
xlim([0.1,21]);
ylim([0,0.012]);

legend('Original Reference', ...
       '10 Hz Filtered Input', ...
       'LSTM Reconstructed Output', ...
       'MLP Reconstructed Output', ...
       'Location', 'northwest');

grid on;

% --------------------------------------------
% Figure Export Settings
% --------------------------------------------

set(gcf, 'Units', 'inches');
set(gcf, 'Position', [1 1 6.5 7]);

set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [6.5 7]);
set(gcf, 'PaperPosition', [0 0 6.5 7]);

print(gcf, 'Filter_vs_LSTM_vs_MLP.png', '-dpng', '-r300');


%% ============================================
% FRF Comparison
% Filtered vs Reference
% LSTM vs Reference
% MLP vs Reference
% ============================================

epsilon = 1e-12;

FRF_filter = P1_cutoffs(:,5) ./ (P1_ref + epsilon);
FRF_lstm   = P1_lstm ./ (P1_ref + epsilon);
FRF_mlp    = P1_mlp ./ (P1_ref + epsilon);

figure;

subplot(2,1,1);

plot(f, abs(FRF_filter), 'LineWidth', 1.2);
hold on;
plot(f, abs(FRF_lstm), 'LineWidth', 1.2);
plot(f, abs(FRF_mlp), 'LineWidth', 1.2);
hold off;

xlabel('frequency (Hz)');
ylabel('FRF magnitude');
title('FRF Comparison: 0-20 Hz');
xlim([0,20]);
ylim([0,20]);

legend('10 Hz Filtered / Reference', ...
       'LSTM / Reference', ...
       'MLP / Reference', ...
       'Location', 'northeast');

grid on;

subplot(2,1,2);

plot(f, abs(FRF_filter), 'LineWidth', 1.2);
hold on;
plot(f, abs(FRF_lstm), 'LineWidth', 1.2);
plot(f, abs(FRF_mlp), 'LineWidth', 1.2);
hold off;

xlabel('frequency (Hz)');
ylabel('FRF magnitude');
title('FRF Comparison: 5-20 Hz');
xlim([5,20]);

legend('10 Hz Filtered / Reference', ...
       'LSTM / Reference', ...
       'MLP / Reference', ...
       'Location', 'southeast');

grid on;
% --------------------------------------------
% Figure Export Settings
% --------------------------------------------

set(gcf, 'Units', 'inches');
set(gcf, 'Position', [1 1 6.5 7]);

set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [6.5 7]);
set(gcf, 'PaperPosition', [0 0 6.5 7]);

print(gcf, 'FRF_Filter_vs_LSTM_vs_MLP.png', '-dpng', '-r300');

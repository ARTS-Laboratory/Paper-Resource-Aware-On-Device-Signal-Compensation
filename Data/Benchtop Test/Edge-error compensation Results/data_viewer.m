%% ============================================================
% Plot Reference vs 10 Hz HPF vs LSTM Compensated Signal
% Time Domain, Frequency Domain, and FRF
% ============================================================


filePath = 'C005.CSV';

data = readmatrix(filePath, 'NumHeaderLines', 1);

time_us = data(:,1);
reference_accel = data(:,2);
lstm_accel = data(:,3);

% Convert time to seconds
time = time_us / 1e6;

% Remove mean
reference_accel = reference_accel - mean(reference_accel);
lstm_accel = lstm_accel - mean(lstm_accel);

% Sampling frequency
dt = mean(diff(time));
Fs = 1/dt;
N = length(time);

fprintf('Sampling Frequency = %.4f Hz\n', Fs);

%% ============================================================
% Apply 10 Hz High-Pass Filter to Reference
% ============================================================

filterOrder = 5;
cutoffFreq = 10; % Hz

Wn = cutoffFreq / (Fs/2);

if Wn >= 1
    error('Cutoff frequency must be less than Nyquist frequency.');
end

[b,a] = butter(filterOrder, Wn, 'high');

hpf_accel = filtfilt(b, a, reference_accel);

%% ============================================================
% FFT
% ============================================================

[f, P1_ref] = computeFFT(reference_accel, Fs, N);
[~, P1_hpf] = computeFFT(hpf_accel, Fs, N);
[~, P1_lstm] = computeFFT(lstm_accel, Fs, N);

%% ============================================================
% FRF
% ============================================================

epsilon = 1e-12;

FRF_hpf = P1_hpf ./ (P1_ref + epsilon);
FRF_lstm = P1_lstm ./ (P1_ref + epsilon);

%% ============================================================
% Plot
% ============================================================

figure;

subplot(3,1,1);
plot(time, reference_accel, 'LineWidth', 1.2); hold on;
plot(time, hpf_accel, 'LineWidth', 1.2);
plot(time, lstm_accel, 'LineWidth', 1.2);
hold off;

xlabel('Time (s)');
ylabel('Acceleration');
title('Time Domain');
legend('Reference', '10 Hz HPF', 'LSTM Compensated', ...
       'Location', 'northwest');
grid on;

subplot(3,1,2);
plot(f, P1_ref, 'LineWidth', 1.2); hold on;
plot(f, P1_hpf, 'LineWidth', 1.2);
plot(f, P1_lstm, 'LineWidth', 1.2);
hold off;

xlabel('Frequency (Hz)');
ylabel('Amplitude');
title('Frequency Domain');
legend('Reference', '10 Hz HPF', 'LSTM Compensated', ...
       'Location', 'northwest');
grid on;
xlim([0.1 21]);

subplot(3,1,3);
plot(f, abs(FRF_hpf), 'LineWidth', 1.2); hold on;
plot(f, abs(FRF_lstm), 'LineWidth', 1.2);
hold off;

xlabel('Frequency (Hz)');
ylabel('FRF Magnitude');
title('FRF Comparison');
legend('10 Hz HPF / Reference', ...
       'LSTM Compensated / Reference', ...
       'Location', 'northwest');
grid on;
xlim([0.1 21]);
ylim([0 5]);

%% ============================================================
% Export Figure
% ============================================================

set(gcf, 'Units', 'inches');
set(gcf, 'Position', [1 1 6.5 8]);

set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [6.5 8]);
set(gcf, 'PaperPosition', [0 0 6.5 8]);

print(gcf, 'Reference_HPF_LSTM_Comparison.png', '-dpng', '-r300');
print(gcf, 'Reference_HPF_LSTM_Comparison.pdf', '-dpdf', '-bestfit');

%% ============================================================
% Local Function
% ============================================================

function [f, P1] = computeFFT(signal, Fs, N)

    signal = signal(:);

    Y = fft(signal);
    P2 = abs(Y/N);

    P1 = P2(1:floor(N/2)+1);
    P1(2:end-1) = 2*P1(2:end-1);

    f = Fs*(0:floor(N/2))/N;

end
%% ============================================================
%  Edge LSTM Error Compensator
%  Simplified Paper Plot Script
%  Time column unit: microseconds
% ============================================================

clc; clear; close all;

%% ---------------- User Settings ----------------

trainFile = 'C:\Users\joud\Dropbox\Satme_2026_IEEE_Edge_Processing_Compensator\Data\Benchtop Test\dataset_collection\Color SD Card (Top Sensor Package)\014.csv';
testFile  = 'C:\Users\joud\Dropbox\Satme_2026_IEEE_Edge_Processing_Compensator\Data\Benchtop Test\dataset_collection\Color SD Card (Top Sensor Package)\006.csv';

outDir = 'Paper_Plots';
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

filterOrder = 5;
selectedCutoff = 10;      % Hz

numHiddenUnits = 100;
maxEpochs = 100000;
miniBatchSize = 100;
initialLearnRate = 0.001;

%% ---------------- Load Training Data ----------------

trainData = readmatrix(trainFile, 'NumHeaderLines', 1);

timeTrain = trainData(:,1) / 1e6;   % microseconds to seconds
timeTrain = timeTrain - timeTrain(1);

accTrain = trainData(:,2);
accTrain = accTrain - mean(accTrain);

dtTrain = mean(diff(timeTrain), 'omitnan');
FsTrain = 1/dtTrain;
NTrain  = length(accTrain);

%% ---------------- Load Testing Data ----------------

testData = readmatrix(testFile, 'NumHeaderLines', 1);

timeTest = testData(:,1) / 1e6;     % microseconds to seconds
timeTest = timeTest - timeTest(1);

accTest = testData(:,2);
accTest = accTest - mean(accTest);

dtTest = mean(diff(timeTest), 'omitnan');
FsTest = 1/dtTest;
NTest  = length(accTest);

fprintf('FsTrain = %.4f Hz\n', FsTrain);
fprintf('FsTest  = %.4f Hz\n', FsTest);
fprintf('Nyquist Train = %.4f Hz\n', FsTrain/2);
fprintf('Nyquist Test  = %.4f Hz\n', FsTest/2);

assert(selectedCutoff < FsTrain/2, 'Cutoff exceeds training Nyquist frequency.');
assert(selectedCutoff < FsTest/2,  'Cutoff exceeds testing Nyquist frequency.');

%% ---------------- Apply 10 Hz High-Pass Filter ----------------

WnTrain = selectedCutoff/(FsTrain/2);
[bTrain,aTrain] = butter(filterOrder, WnTrain, 'high');
inputTrain = filtfilt(bTrain, aTrain, accTrain);

WnTest = selectedCutoff/(FsTest/2);
[bTest,aTest] = butter(filterOrder, WnTest, 'high');
inputTest = filtfilt(bTest, aTest, accTest);

%% ---------------- FFT Data ----------------

computeFFT = @(x,Fs,N) deal(Fs*(0:floor(N/2))/N, singleSidedFFT(x,N));

[fTrain, P1_train_ref]   = computeFFT(accTrain, FsTrain, NTrain);
[~,      P1_train_input] = computeFFT(inputTrain, FsTrain, NTrain);

[fTest, P1_test_ref]   = computeFFT(accTest, FsTest, NTest);
[~,     P1_test_input] = computeFFT(inputTest, FsTest, NTest);

%% ============================================================
%  Figure 1: Training Data Time and Frequency Domain
% ============================================================

figure;

subplot(2,1,1);
plot(timeTrain, accTrain, 'LineWidth', 1.2); hold on;
plot(timeTrain, inputTrain, 'LineWidth', 1.2);
hold off;
xlabel('Time (s)');
ylabel('Acceleration');
title('Training Data: Time Domain');
legend('Reference', '10 Hz HPF', 'Location', 'northwest');
grid on;
xlim([min(timeTrain), max(timeTrain)]);

subplot(2,1,2);
plot(fTrain, P1_train_ref, 'LineWidth', 1.2); hold on;
plot(fTrain, P1_train_input, 'LineWidth', 1.2);
hold off;
xlabel('Frequency (Hz)');
ylabel('Amplitude');
title('Training Data: Frequency Domain');
legend('Reference', '10 Hz HPF', 'Location', 'northwest');
grid on;
xlim([0.1,21]);

exportFigure(gcf, fullfile(outDir, 'Fig_01_Training_Time_Frequency'));

%% ============================================================
%  Figure 2: Testing Data Time and Frequency Domain
% ============================================================

figure;

subplot(2,1,1);
plot(timeTest, accTest, 'LineWidth', 1.2); hold on;
plot(timeTest, inputTest, 'LineWidth', 1.2);
hold off;
xlabel('Time (s)');
ylabel('Acceleration');
title('Testing Data: Time Domain');
legend('Reference', '10 Hz HPF', 'Location', 'northwest');
grid on;
xlim([min(timeTest), max(timeTest)]);

subplot(2,1,2);
plot(fTest, P1_test_ref, 'LineWidth', 1.2); hold on;
plot(fTest, P1_test_input, 'LineWidth', 1.2);
hold off;
xlabel('Frequency (Hz)');
ylabel('Amplitude');
title('Testing Data: Frequency Domain');
legend('Reference', '10 Hz HPF', 'Location', 'northwest');
grid on;
xlim([0.1,21]);

exportFigure(gcf, fullfile(outDir, 'Fig_02_Testing_Time_Frequency'));

%% ---------------- Normalize Training Data ----------------

inputMu  = mean(inputTrain);
inputSig = std(inputTrain);

targetMu  = mean(accTrain);
targetSig = std(accTrain);

inputTrainNorm  = (inputTrain - inputMu) / inputSig;
targetTrainNorm = (accTrain   - targetMu) / targetSig;

XTrain = {inputTrainNorm'};
YTrain = {targetTrainNorm'};

%% ---------------- Define LSTM Network ----------------

layers = [
    sequenceInputLayer(1)
    lstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
    fullyConnectedLayer(1)
    regressionLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'InitialLearnRate', initialLearnRate, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 100, ...
    'GradientThreshold', 1, ...
    'Shuffle', 'never', ...
    'Plots', 'training-progress', ...
    'Verbose', true);

%% ---------------- Train LSTM ----------------

net = trainNetwork(XTrain, YTrain, layers, options);

%% ---------------- Predict on Testing Data ----------------

inputTestNorm = (inputTest - inputMu) / inputSig;
XTest = {inputTestNorm'};

YPredNorm = predict(net, XTest, 'MiniBatchSize', 1);
YPredNorm = YPredNorm{1}';

accLSTM = YPredNorm * targetSig + targetMu;

[~, P1_lstm] = computeFFT(accLSTM, FsTest, NTest);

%% ============================================================
%  Figure 3: LSTM Compensation Time and Frequency Domain
% ============================================================

figure;

subplot(2,1,1);
plot(timeTest, accTest, 'LineWidth', 1.2); hold on;
plot(timeTest, inputTest, 'LineWidth', 1.2);
plot(timeTest, accLSTM, 'LineWidth', 1.2);
hold off;
xlabel('Time (s)');
ylabel('Acceleration');
title('LSTM Compensation: Time Domain');
legend('Reference', '10 Hz HPF', 'LSTM Compensated', ...
       'Location', 'northwest');
grid on;
xlim([min(timeTest), max(timeTest)]);

subplot(2,1,2);
plot(fTest, P1_test_ref, 'LineWidth', 1.2); hold on;
plot(fTest, P1_test_input, 'LineWidth', 1.2);
plot(fTest, P1_lstm, 'LineWidth', 1.2);
hold off;
xlabel('Frequency (Hz)');
ylabel('Amplitude');
title('LSTM Compensation: Frequency Domain');
legend('Reference', '10 Hz HPF', 'LSTM Compensated', ...
       'Location', 'northwest');
grid on;
xlim([0.1,21]);

exportFigure(gcf, fullfile(outDir, 'Fig_03_LSTM_Time_Frequency'));

%% ============================================================
%  Figure 4: FRF Comparison
% ============================================================

epsilon = 1e-12;

FRF_input = P1_test_input ./ (P1_test_ref + epsilon);
FRF_lstm  = P1_lstm       ./ (P1_test_ref + epsilon);

figure;

subplot(2,1,1);
plot(fTest, abs(FRF_input), 'LineWidth', 1.2); hold on;
plot(fTest, abs(FRF_lstm), 'LineWidth', 1.2);
hold off;
xlabel('Frequency (Hz)');
ylabel('FRF Magnitude');
title('FRF Comparison: 0--20 Hz');
legend('10 Hz HPF / Reference', 'LSTM / Reference', ...
       'Location', 'northwest');
grid on;
xlim([0,20]);
ylim([0,20]);

subplot(2,1,2);
plot(fTest, abs(FRF_input), 'LineWidth', 1.2); hold on;
plot(fTest, abs(FRF_lstm), 'LineWidth', 1.2);
hold off;
xlabel('Frequency (Hz)');
ylabel('FRF Magnitude');
title('FRF Comparison: 5--20 Hz');
legend('10 Hz HPF / Reference', 'LSTM / Reference', ...
       'Location', 'southeast');
grid on;
xlim([5,20]);

exportFigure(gcf, fullfile(outDir, 'Fig_04_FRF_Filter_vs_LSTM'));

%% ============================================================
%  Figure 5: Time-Domain Error Plot
% ============================================================

errorInput = accTest - inputTest;
errorLSTM  = accTest - accLSTM;

figure;

plot(timeTest, errorInput, 'LineWidth', 1.2); hold on;
plot(timeTest, errorLSTM, 'LineWidth', 1.2);
hold off;
xlabel('Time (s)');
ylabel('Error');
title('Time-Domain Reconstruction Error');
legend('10 Hz HPF Error', 'LSTM Error', 'Location', 'northwest');
grid on;
xlim([min(timeTest), max(timeTest)]);

exportFigure(gcf, fullfile(outDir, 'Fig_05_Time_Domain_Error'));

%% ============================================================
%  Metrics
% ============================================================

rmseInput = sqrt(mean((accTest - inputTest).^2));
rmseLSTM  = sqrt(mean((accTest - accLSTM).^2));

snrInput = 10*log10(sum(accTest.^2) / sum((accTest - inputTest).^2));
snrLSTM  = 10*log10(sum(accTest.^2) / sum((accTest - accLSTM).^2));

tracInput = (accTest' * inputTest)^2 / ...
            ((accTest' * accTest) * (inputTest' * inputTest));

tracLSTM = (accTest' * accLSTM)^2 / ...
           ((accTest' * accTest) * (accLSTM' * accLSTM));

fprintf('\n================ Performance Metrics ================\n');
fprintf('RMSE - 10 Hz HPF Input : %.6f\n', rmseInput);
fprintf('RMSE - LSTM Output     : %.6f\n', rmseLSTM);
fprintf('SNR  - 10 Hz HPF Input : %.6f dB\n', snrInput);
fprintf('SNR  - LSTM Output     : %.6f dB\n', snrLSTM);
fprintf('TRAC - 10 Hz HPF Input : %.6f\n', tracInput);
fprintf('TRAC - LSTM Output     : %.6f\n', tracLSTM);

metricsTable = table(rmseInput, rmseLSTM, snrInput, snrLSTM, ...
                     tracInput, tracLSTM);

writetable(metricsTable, fullfile(outDir, 'LSTM_Performance_Metrics.csv'));

%% ============================================================
%  Export Trained LSTM Network for Teensy 4.0 Deployment
%  Short 8.3-compatible filenames
% ============================================================

disp(' ');
disp('====================================================');
disp('Exporting Trained LSTM Parameters for Teensy 4.0');
disp('====================================================');

exportDir = fullfile(outDir, 'Teensy_LSTM_Parameters');

if ~exist(exportDir, 'dir')
    mkdir(exportDir);
end

save(fullfile(exportDir, 'trained_LSTM_compensator.mat'), ...
     'net', 'inputMu', 'inputSig', 'targetMu', 'targetSig');

layersNet = net.Layers;

lstmLayerObj = layersNet(2);
fcLayerObj   = layersNet(3);

InputWeights     = lstmLayerObj.InputWeights;
RecurrentWeights = lstmLayerObj.RecurrentWeights;
Bias             = lstmLayerObj.Bias;

FCWeights = fcLayerObj.Weights;
FCBias    = fcLayerObj.Bias;

writematrix(InputWeights,     fullfile(exportDir, 'WI.CSV'));
writematrix(RecurrentWeights, fullfile(exportDir, 'WH.CSV'));
writematrix(Bias,             fullfile(exportDir, 'B.CSV'));
writematrix(FCWeights,        fullfile(exportDir, 'FCW.CSV'));
writematrix(FCBias,           fullfile(exportDir, 'FCB.CSV'));

normalizationData = [
    inputMu
    inputSig
    targetMu
    targetSig
];

writematrix(normalizationData, fullfile(exportDir, 'NORM.CSV'));

metadata = [
    numHiddenUnits
    selectedCutoff
    filterOrder
    FsTrain
    FsTest
];

writematrix(metadata, fullfile(exportDir, 'META.CSV'));

disp(' ');
disp('Export Complete');
disp(' ');

fprintf('InputWeights      : %d x %d\n', size(InputWeights,1), size(InputWeights,2));
fprintf('RecurrentWeights  : %d x %d\n', size(RecurrentWeights,1), size(RecurrentWeights,2));
fprintf('Bias              : %d x %d\n', size(Bias,1), size(Bias,2));
fprintf('FCWeights         : %d x %d\n', size(FCWeights,1), size(FCWeights,2));
fprintf('FCBias            : %d x %d\n', size(FCBias,1), size(FCBias,2));

disp('Files Written:');
disp(fullfile(exportDir, 'WI.CSV'));
disp(fullfile(exportDir, 'WH.CSV'));
disp(fullfile(exportDir, 'B.CSV'));
disp(fullfile(exportDir, 'FCW.CSV'));
disp(fullfile(exportDir, 'FCB.CSV'));
disp(fullfile(exportDir, 'NORM.CSV'));
disp(fullfile(exportDir, 'META.CSV'));
disp(fullfile(exportDir, 'trained_LSTM_compensator.mat'));

disp('Script completed successfully.');

% ============================================================
%  Local Functions
% ============================================================

function P1 = singleSidedFFT(signal,N)

    signal = signal(:);

    Y = fft(signal);
    P2 = abs(Y/N);

    P1 = P2(1:floor(N/2)+1);
    P1(2:end-1) = 2*P1(2:end-1);

end

function exportFigure(figHandle, fileName)

    set(figHandle, 'Units', 'inches');
    set(figHandle, 'Position', [1 1 6.5 7]);

    set(figHandle, 'PaperUnits', 'inches');
    set(figHandle, 'PaperSize', [6.5 7]);
    set(figHandle, 'PaperPosition', [0 0 6.5 7]);

    print(figHandle, [fileName '.png'], '-dpng', '-r300');
    print(figHandle, [fileName '.pdf'], '-dpdf', '-bestfit');

end
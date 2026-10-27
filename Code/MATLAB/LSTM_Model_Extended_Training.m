%% ============================================================
%  Edge LSTM Error Compensator
%  Multiple-File Training Version
%
%  Main template:
%       LSTM_Model_Training.m
%
%  Training dataset:
%       Same training and testing files used in
%       "Shaker Data Graphing Code.py"
%
%  The major change is that multiple independent training files are loaded,
%  filtered, normalized together, and converted into sliding windows.
%  Windows never cross file boundaries.
%
%  Time column unit: microseconds
%% ============================================================

clc;
clear;
close all;

%% ---------------- User Settings ----------------

originalDataFolder = [ ...
    'C:\Users\joud\Dropbox\' ...
    'Satme_2026_IEEE_Edge_Processing_Compensator\Data\' ...
    'Benchtop Test\dataset_collection\' ...
    'Color SD Card (Top Sensor Package)'];

compensationDataFolder = ['C:\Users\joud\Dropbox\Satme_2026_IEEE_Edge_Processing_Compensator\Data\Benchtop Test\Edge-error compensation Results'];

trainFiles = { ...
    fullfile(originalDataFolder, '006.csv')
    fullfile(originalDataFolder, '008.csv')
    fullfile(originalDataFolder, '009.csv')
    fullfile(originalDataFolder, '010.csv')
    fullfile(originalDataFolder, '011.csv')
    fullfile(originalDataFolder, '012.csv')
    fullfile(originalDataFolder, '013.csv')
    fullfile(compensationDataFolder, 'C006.CSV')
    fullfile(compensationDataFolder, 'C007.CSV')
    fullfile(compensationDataFolder, 'C008.CSV')
    fullfile(compensationDataFolder, 'C009.CSV')
    fullfile(compensationDataFolder, 'C010.CSV')
    };

testFile = fullfile(originalDataFolder, '014.csv');

outDir = 'Model_Multiple_Training_Files';

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

%% ---------------- Signal Processing Settings ----------------

filterOrder = 5;
selectedCutoff = 10;      % Hz

%% ---------------- LSTM Settings ----------------

numHiddenUnits = 175;
maxEpochs = 20;
miniBatchSize = 32;
initialLearnRate = 0.001;

% Number of previous samples supplied to the LSTM.
windowSize = 400;

% Use 1 to reproduce the Python sliding-window procedure.
% Increase this value only if memory consumption becomes too high.
trainingStride = 200;
testingStride = 200;

%% ---------------- Validate Input Files ----------------

for fileIndex = 1:numel(trainFiles)

    if ~isfile(trainFiles{fileIndex})
        error('Training file not found:\n%s', trainFiles{fileIndex});
    end
end

if ~isfile(testFile)
    error('Testing file not found:\n%s', testFile);
end

fprintf('\n====================================================\n');
fprintf('MULTIPLE-FILE LSTM TRAINING\n');
fprintf('====================================================\n');
fprintf('Training files : %d\n', numel(trainFiles));
fprintf('Testing file   : %s\n', testFile);
fprintf('Window size    : %d samples\n', windowSize);
fprintf('Training stride: %d\n', trainingStride);

%% ============================================================
%  Load and Process All Training Files
%% ============================================================

numberOfTrainingFiles = numel(trainFiles);

trainingData = repmat(struct( ...
    'File', "", ...
    'Time', [], ...
    'Reference', [], ...
    'Filtered', [], ...
    'SamplingRate', [], ...
    'NumberOfSamples', []), ...
    numberOfTrainingFiles, 1);

allFilteredTraining = [];
allReferenceTraining = [];

fprintf('\nLoading and filtering training files...\n');

for fileIndex = 1:numberOfTrainingFiles

    currentFile = trainFiles{fileIndex};

    currentData = readmatrix(currentFile, 'NumHeaderLines', 1);

    if size(currentData,2) < 2
        error('Training file must contain time and acceleration:\n%s', ...
            currentFile);
    end

    validRows = ...
        isfinite(currentData(:,1)) & ...
        isfinite(currentData(:,2));

    currentData = currentData(validRows,:);

    timeCurrent = currentData(:,1) / 1e6;
    timeCurrent = timeCurrent - timeCurrent(1);

    referenceCurrent = currentData(:,2);
    referenceCurrent = ...
        referenceCurrent - mean(referenceCurrent, 'omitnan');

    dtCurrent = mean(diff(timeCurrent), 'omitnan');
    FsCurrent = 1 / dtCurrent;

    if selectedCutoff >= FsCurrent/2
        error(['The cutoff frequency exceeds the Nyquist frequency ' ...
            'for:\n%s'], currentFile);
    end

    normalizedCutoff = selectedCutoff / (FsCurrent/2);

    [bCurrent, aCurrent] = butter( ...
        filterOrder, ...
        normalizedCutoff, ...
        'high');

    filteredCurrent = filtfilt( ...
        bCurrent, ...
        aCurrent, ...
        referenceCurrent);

    filteredCurrent = ...
        filteredCurrent - mean(filteredCurrent, 'omitnan');

    trainingData(fileIndex).File = string(currentFile);
    trainingData(fileIndex).Time = timeCurrent;
    trainingData(fileIndex).Reference = referenceCurrent;
    trainingData(fileIndex).Filtered = filteredCurrent;
    trainingData(fileIndex).SamplingRate = FsCurrent;
    trainingData(fileIndex).NumberOfSamples = ...
        numel(referenceCurrent);

    % Concatenate only to compute common normalization statistics.
    allFilteredTraining = [ ...
        allFilteredTraining; ...
        filteredCurrent]; %#ok<AGROW>

    allReferenceTraining = [ ...
        allReferenceTraining; ...
        referenceCurrent]; %#ok<AGROW>

    fprintf('  %2d/%2d: %-10s | N = %7d | Fs = %.4f Hz\n', ...
        fileIndex, ...
        numberOfTrainingFiles, ...
        getFileName(currentFile), ...
        numel(referenceCurrent), ...
        FsCurrent);
end

%% ---------------- Global Normalization ----------------
%
% These normalization statistics are fit using all training files,
% matching the Python StandardScaler workflow.

inputMu = mean(allFilteredTraining, 'omitnan');
inputSig = std(allFilteredTraining, 0, 'omitnan');

targetMu = mean(allReferenceTraining, 'omitnan');
targetSig = std(allReferenceTraining, 0, 'omitnan');

if inputSig == 0 || targetSig == 0
    error('A calculated normalization standard deviation is zero.');
end

fprintf('\nGlobal normalization values:\n');
fprintf('Input mean  : %.10g\n', inputMu);
fprintf('Input std   : %.10g\n', inputSig);
fprintf('Target mean : %.10g\n', targetMu);
fprintf('Target std  : %.10g\n', targetSig);

%% ============================================================
%  Build Multi-File Sliding-Window Training Dataset
%% ============================================================
%
% Each training example contains:
%
%   Input  = windowSize high-pass-filtered samples
%   Target = reference sample immediately after the window
%
% Each file is processed independently, so no training window crosses
% from the end of one experiment into the beginning of another.

numberOfWindowsPerFile = zeros(numberOfTrainingFiles,1);

for fileIndex = 1:numberOfTrainingFiles

    currentLength = trainingData(fileIndex).NumberOfSamples;

    numberOfWindowsPerFile(fileIndex) = ...
        numel(windowSize+1 : trainingStride : currentLength);
end

totalTrainingWindows = sum(numberOfWindowsPerFile);

fprintf('\nTotal training windows: %d\n', totalTrainingWindows);

XTrain = cell(totalTrainingWindows,1);
YTrain = zeros(totalTrainingWindows,1);

windowCounter = 0;

fprintf('Constructing training windows...\n');

for fileIndex = 1:numberOfTrainingFiles

    filteredCurrent = trainingData(fileIndex).Filtered;
    referenceCurrent = trainingData(fileIndex).Reference;

    filteredNormalized = ...
        (filteredCurrent-inputMu) / inputSig;

    referenceNormalized = ...
        (referenceCurrent-targetMu) / targetSig;

    targetIndices = ...
        windowSize+1 : trainingStride : numel(filteredNormalized);

    for targetIndex = targetIndices

        windowCounter = windowCounter + 1;

        startIndex = targetIndex-windowSize;
        endIndex = targetIndex-1;

        % One feature by windowSize time steps.
        XTrain{windowCounter} = ...
            filteredNormalized(startIndex:endIndex).';

        YTrain(windowCounter) = ...
            referenceNormalized(targetIndex);
    end

    fprintf('  %s: %d windows\n', ...
        getFileName(trainingData(fileIndex).File), ...
        numel(targetIndices));
end

%% ============================================================
%  Load and Process Testing Data
%% ============================================================

testData = readmatrix(testFile, 'NumHeaderLines', 1);

if size(testData,2) < 2
    error('Testing file must contain time and acceleration columns.');
end

validTestRows = ...
    isfinite(testData(:,1)) & ...
    isfinite(testData(:,2));

testData = testData(validTestRows,:);

timeTest = testData(:,1) / 1e6;
timeTest = timeTest-timeTest(1);

accTest = testData(:,2);
accTest = accTest-mean(accTest, 'omitnan');

dtTest = mean(diff(timeTest), 'omitnan');
FsTest = 1/dtTest;
NTest = numel(accTest);

assert(selectedCutoff < FsTest/2, ...
    'Cutoff exceeds testing Nyquist frequency.');

WnTest = selectedCutoff/(FsTest/2);

[bTest,aTest] = butter( ...
    filterOrder, ...
    WnTest, ...
    'high');

inputTest = filtfilt( ...
    bTest, ...
    aTest, ...
    accTest);

inputTest = inputTest-mean(inputTest, 'omitnan');

%% ---------------- Build Testing Windows ----------------

inputTestNorm = ...
    (inputTest-inputMu) / inputSig;

targetTestNorm = ...
    (accTest-targetMu) / targetSig;

testTargetIndices = ...
    windowSize+1 : testingStride : NTest;

numberOfTestWindows = numel(testTargetIndices);

XTest = cell(numberOfTestWindows,1);
YTest = zeros(numberOfTestWindows,1);

for testWindowIndex = 1:numberOfTestWindows

    targetIndex = testTargetIndices(testWindowIndex);

    startIndex = targetIndex-windowSize;
    endIndex = targetIndex-1;

    XTest{testWindowIndex} = ...
        inputTestNorm(startIndex:endIndex).';

    YTest(testWindowIndex) = ...
        targetTestNorm(targetIndex);
end

timePrediction = timeTest(testTargetIndices);
referencePrediction = accTest(testTargetIndices);
filteredPrediction = inputTest(testTargetIndices);

fprintf('\nTesting samples : %d\n', NTest);
fprintf('Testing Fs      : %.4f Hz\n', FsTest);
fprintf('Testing windows : %d\n', numberOfTestWindows);

%% ============================================================
%  Define Sequence-to-One LSTM Network
%% ============================================================
%
% The Python model predicts one reference value from each 400-sample
% sequence, so OutputMode is set to "last".

layers = [
    sequenceInputLayer(1)
    lstmLayer(numHiddenUnits, 'OutputMode', 'last')
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
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true);

%% ---------------- Train LSTM ----------------

trainingTimer = tic;

net = trainNetwork( ...
    XTrain, ...
    YTrain, ...
    layers, ...
    options);

trainingElapsedTime = toc(trainingTimer);

fprintf('\nTraining completed in %.3f seconds.\n', ...
    trainingElapsedTime);

%% ---------------- Predict Testing Data ----------------

predictionTimer = tic;

YPredNorm = predict( ...
    net, ...
    XTest, ...
    'MiniBatchSize', miniBatchSize);

predictionElapsedTime = toc(predictionTimer);

YPredNorm = YPredNorm(:);

accLSTM = ...
    YPredNorm*targetSig + targetMu;

fprintf('Prediction completed in %.3f seconds.\n', ...
    predictionElapsedTime);

%% ---------------- FFT Data ----------------

[fPrediction,P1PredictionReference] = ...
    computeSingleSidedFFT(referencePrediction,FsTest);

[~,P1PredictionFiltered] = ...
    computeSingleSidedFFT(filteredPrediction,FsTest);

[~,P1LSTM] = ...
    computeSingleSidedFFT(accLSTM,FsTest);

%% ============================================================
%  Figure: LSTM Compensation
%% ============================================================

figure;

subplot(2,1,1);

plot(timePrediction,referencePrediction,'LineWidth',1.2);
hold on;
plot(timePrediction,filteredPrediction,'LineWidth',1.2);
plot(timePrediction,accLSTM,'LineWidth',1.2);
hold off;

xlabel('Time (s)');
ylabel('Acceleration');
title('Multiple-File LSTM Compensation: Time Domain');

legend( ...
    'Reference', ...
    '10 Hz HPF', ...
    'LSTM Compensated', ...
    'Location','northwest');

grid on;
xlim([min(timePrediction),max(timePrediction)]);

subplot(2,1,2);

plot(fPrediction,P1PredictionReference,'LineWidth',1.2);
hold on;
plot(fPrediction,P1PredictionFiltered,'LineWidth',1.2);
plot(fPrediction,P1LSTM,'LineWidth',1.2);
hold off;

xlabel('Frequency (Hz)');
ylabel('Amplitude');
title('Multiple-File LSTM Compensation: Frequency Domain');

legend( ...
    'Reference', ...
    '10 Hz HPF', ...
    'LSTM Compensated', ...
    'Location','northwest');

grid on;
xlim([0.1 21]);

exportFigure( ...
    gcf, ...
    fullfile(outDir,'Fig_01_LSTM_Time_Frequency'));

%% ============================================================
%  FRF Comparison
%% ============================================================

referenceFFT = fft(referencePrediction);
filteredFFT = fft(filteredPrediction);
lstmFFT = fft(accLSTM);

numberOfPredictionSamples = numel(referencePrediction);

frfFrequency = ...
    FsTest*(0:floor(numberOfPredictionSamples/2))' / ...
    numberOfPredictionSamples;

singleSidedIndices = ...
    1:floor(numberOfPredictionSamples/2)+1;

referenceFFTSingle = referenceFFT(singleSidedIndices);
filteredFFTSingle = filteredFFT(singleSidedIndices);
lstmFFTSingle = lstmFFT(singleSidedIndices);

referenceThreshold = ...
    0.01*max(abs(referenceFFTSingle));

validFRFBins = ...
    abs(referenceFFTSingle) > referenceThreshold;

FRFFiltered = zeros(size(referenceFFTSingle));
FRFLSTM = zeros(size(referenceFFTSingle));

FRFFiltered(validFRFBins) = ...
    filteredFFTSingle(validFRFBins) ./ ...
    referenceFFTSingle(validFRFBins);

FRFLSTM(validFRFBins) = ...
    lstmFFTSingle(validFRFBins) ./ ...
    referenceFFTSingle(validFRFBins);

figure;

plot(frfFrequency,abs(FRFFiltered),'LineWidth',1.2);
hold on;
plot(frfFrequency,abs(FRFLSTM),'LineWidth',1.2);
hold off;

xlabel('Frequency (Hz)');
ylabel('Amplitude Ratio');
title('FRF Magnitude Comparison');

legend( ...
    '10 Hz HPF / Reference', ...
    'LSTM / Reference', ...
    'Location','northwest');

grid on;
xlim([0.1 21]);

exportFigure( ...
    gcf, ...
    fullfile(outDir,'Fig_02_FRF_Comparison'));

%% ============================================================
%  Performance Metrics
%% ============================================================

rmseInput = sqrt(mean( ...
    (referencePrediction-filteredPrediction).^2, ...
    'omitnan'));

rmseLSTM = sqrt(mean( ...
    (referencePrediction-accLSTM).^2, ...
    'omitnan'));

snrInput = 10*log10( ...
    sum(referencePrediction.^2) / ...
    sum((referencePrediction-filteredPrediction).^2));

snrLSTM = 10*log10( ...
    sum(referencePrediction.^2) / ...
    sum((referencePrediction-accLSTM).^2));

tracInput = ...
    (referencePrediction'*filteredPrediction)^2 / ...
    ((referencePrediction'*referencePrediction) * ...
     (filteredPrediction'*filteredPrediction));

tracLSTM = ...
    (referencePrediction'*accLSTM)^2 / ...
    ((referencePrediction'*referencePrediction) * ...
     (accLSTM'*accLSTM));

fprintf('\n================ Performance Metrics ================\n');
fprintf('RMSE - 10 Hz HPF Input : %.6f\n',rmseInput);
fprintf('RMSE - LSTM Output     : %.6f\n',rmseLSTM);
fprintf('SNR  - 10 Hz HPF Input : %.6f dB\n',snrInput);
fprintf('SNR  - LSTM Output     : %.6f dB\n',snrLSTM);
fprintf('TRAC - 10 Hz HPF Input : %.6f\n',tracInput);
fprintf('TRAC - LSTM Output     : %.6f\n',tracLSTM);

metricsTable = table( ...
    rmseInput, ...
    rmseLSTM, ...
    snrInput, ...
    snrLSTM, ...
    tracInput, ...
    tracLSTM);

writetable( ...
    metricsTable, ...
    fullfile(outDir,'LSTM_Performance_Metrics.csv'));

%% ============================================================
%  Export Trained LSTM Network for Teensy 4.0
%% ============================================================

exportDir = fullfile(outDir,'Teensy_LSTM_Parameters');

if ~exist(exportDir,'dir')
    mkdir(exportDir);
end

save( ...
    fullfile(exportDir,'trained_LSTM_compensator.mat'), ...
    'net', ...
    'inputMu', ...
    'inputSig', ...
    'targetMu', ...
    'targetSig', ...
    'trainFiles', ...
    'testFile', ...
    'windowSize');

layersNet = net.Layers;

lstmLayerObj = layersNet(2);
fcLayerObj = layersNet(3);

InputWeights = lstmLayerObj.InputWeights;
RecurrentWeights = lstmLayerObj.RecurrentWeights;
Bias = lstmLayerObj.Bias;

FCWeights = fcLayerObj.Weights;
FCBias = fcLayerObj.Bias;

writematrix(InputWeights, ...
    fullfile(exportDir,'WI.CSV'));

writematrix(RecurrentWeights, ...
    fullfile(exportDir,'WH.CSV'));

writematrix(Bias, ...
    fullfile(exportDir,'B.CSV'));

writematrix(FCWeights, ...
    fullfile(exportDir,'FCW.CSV'));

writematrix(FCBias, ...
    fullfile(exportDir,'FCB.CSV'));

normalizationData = [ ...
    inputMu
    inputSig
    targetMu
    targetSig
    ];

writematrix(normalizationData, ...
    fullfile(exportDir,'NORM.CSV'));

metadata = [ ...
    numHiddenUnits
    selectedCutoff
    filterOrder
    windowSize
    numberOfTrainingFiles
    FsTest
    ];

writematrix(metadata, ...
    fullfile(exportDir,'META.CSV'));

fprintf('\nExport complete:\n%s\n',exportDir);

disp('Script completed successfully.');

%% ============================================================
%  Local Functions
%% ============================================================

function [frequency,amplitude] = computeSingleSidedFFT(signal,Fs)

    signal = signal(:);
    N = numel(signal);

    Y = fft(signal);
    P2 = abs(Y/N);

    amplitude = P2(1:floor(N/2)+1);

    if rem(N,2) == 0
        amplitude(2:end-1) = ...
            2*amplitude(2:end-1);
    else
        amplitude(2:end) = ...
            2*amplitude(2:end);
    end

    frequency = Fs*(0:floor(N/2))'/N;
end

function exportFigure(figHandle,fileName)

    set(figHandle,'Units','inches');
    set(figHandle,'Position',[1 1 6.5 7]);

    set(figHandle,'PaperUnits','inches');
    set(figHandle,'PaperSize',[6.5 7]);
    set(figHandle,'PaperPosition',[0 0 6.5 7]);

    print(figHandle,[fileName '.png'],'-dpng','-r300');
end

function fileName = getFileName(filePath)

    [~,name,extension] = fileparts(char(filePath));
    fileName = string(name) + string(extension);
end

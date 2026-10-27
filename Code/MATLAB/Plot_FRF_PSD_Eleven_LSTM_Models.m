%% ============================================================
%  Eleven-Model FRF and PSD Comparison
%
%  Compares:
%       10 Hz high-pass-filtered signal
%       1-unit LSTM
%       2-unit LSTM
%       4-unit LSTM
%       8-unit LSTM
%       10-unit LSTM
%       25-unit LSTM
%       50-unit LSTM
%       100-unit LSTM
%       150-unit LSTM
%       170-unit LSTM
%       175-unit LSTM
%
%  Each model folder must contain:
%       Predictions.csv
%       Model_Settings.csv
%
%  Predictions.csv columns:
%       Time
%       Reference
%       Prediction
%
%  FRF calculation:
%       H(f) = FFT(Output) / FFT(Reference)
%
%  No denominator threshold is applied.
%% ============================================================

clc;
clear;
close all;

% ---------------- User Settings ----------------

baseFolder = ['C:\Users\joud\Dropbox\' ...
    'Satme_2026_IEEE_Edge_Processing_Compensator\Data\' ...
    'Benchtop Test\dataset_collection'];

modelFolderNames = { ...
    'Unit1_Epoch20_Batch32_Window400'
    'Unit2_Epoch20_Batch32_Window400'
    'Unit4_Epoch20_Batch32_Window400'
    'Unit8_Epoch20_Batch32_Window400'
    'Unit10_Epoch20_Batch32_Window400'
    'Unit25_Epoch20_Batch32_Window400'
    'Unit50_Epoch20_Batch32_Window400'
    'Unit100_Epoch20_Batch32_Window400'
    'Unit150_Epoch20_Batch32_Window400'
    'Unit170_Epoch20_Batch32_Window400'
    'Unit175_Epoch20_Batch32_Window400'
    };

modelLabels = { ...
    '1-unit LSTM'
    '2-unit LSTM'
    '4-unit LSTM'
    '8-unit LSTM'
    '10-unit LSTM'
    '25-unit LSTM'
    '50-unit LSTM'
    '100-unit LSTM'
    '150-unit LSTM'
    '170-unit LSTM'
    '175-unit LSTM'
    };

expectedHiddenUnits = [1 2 4 8 10 25 50 100 150 170 175];

outDir = fullfile(baseFolder, 'Model_FRF_PSD_Comparison');

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

% Shared signal-processing settings
filterOrder = 3;
highpassCutoff = 10;          % Hz
frequencyLimit = [0.1 25];

% FRF settings
frfScale = "linear";          % "linear" or "db"
movingAverageWindow = 20;     % Frequency bins

% PSD settings
requestedSegmentLength = 4096;
overlapFraction = 0.50;
minimumNFFT = 4096;

% Optional fixed axis limits
frfLinearLimit = [0 5];
frfDBLimit = [-40 20];
psdMagnitudeLimit = [-120 -20];

% ---------------- Load All Model Predictions ----------------

numberOfModels = numel(modelFolderNames);

models = repmat(struct( ...
    'HiddenUnits', [], ...
    'Folder', "", ...
    'Time', [], ...
    'Reference', [], ...
    'Prediction', [], ...
    'SamplingRate', []), ...
    numberOfModels, 1);

for modelIndex = 1:numberOfModels

    modelFolder = fullfile( ...
        baseFolder, ...
        modelFolderNames{modelIndex});

    predictionsFile = fullfile( ...
        modelFolder, ...
        'Predictions.csv');

    settingsFile = fullfile( ...
        modelFolder, ...
        'Model_Settings.csv');

    if ~isfolder(modelFolder)
        error('Model folder not found:\n%s', modelFolder);
    end

    if ~isfile(predictionsFile)
        error('Predictions.csv not found:\n%s', predictionsFile);
    end

    hiddenUnits = expectedHiddenUnits(modelIndex);

    if isfile(settingsFile)

        settingsTable = readtable( ...
            settingsFile, ...
            'VariableNamingRule', ...
            'preserve');

        if ismember( ...
                'Hidden Units', ...
                settingsTable.Properties.VariableNames)

            importedHiddenUnits = ...
                double(settingsTable{1, 'Hidden Units'});

            if importedHiddenUnits ~= expectedHiddenUnits(modelIndex)

                error(['Hidden-unit mismatch in:\n%s\n' ...
                    'Expected %d, but found %d.'], ...
                    settingsFile, ...
                    expectedHiddenUnits(modelIndex), ...
                    importedHiddenUnits);
            end

            hiddenUnits = importedHiddenUnits;
        end
    end

    predictionTable = readtable( ...
        predictionsFile, ...
        'VariableNamingRule', ...
        'preserve');

    requiredColumns = {'Time', 'Reference', 'Prediction'};

    for columnIndex = 1:numel(requiredColumns)

        if ~ismember( ...
                requiredColumns{columnIndex}, ...
                predictionTable.Properties.VariableNames)

            error(['Predictions.csv must contain the column "%s":\n%s'], ...
                requiredColumns{columnIndex}, ...
                predictionsFile);
        end
    end

    time = double(predictionTable.('Time'));
    reference = double(predictionTable.('Reference'));
    prediction = double(predictionTable.('Prediction'));

    validRows = ...
        isfinite(time) & ...
        isfinite(reference) & ...
        isfinite(prediction);

    time = time(validRows);
    reference = reference(validRows);
    prediction = prediction(validRows);

    if numel(time) < 2
        error('Not enough valid samples in:\n%s', predictionsFile);
    end

    time = time-time(1);

    reference = ...
        reference-mean(reference, 'omitnan');

    prediction = ...
        prediction-mean(prediction, 'omitnan');

    samplePeriod = mean(diff(time), 'omitnan');

    if ~isfinite(samplePeriod) || samplePeriod <= 0
        error('Invalid time data in:\n%s', predictionsFile);
    end

    models(modelIndex).HiddenUnits = hiddenUnits;
    models(modelIndex).Folder = string(modelFolder);
    models(modelIndex).Time = time;
    models(modelIndex).Reference = reference;
    models(modelIndex).Prediction = prediction;
    models(modelIndex).SamplingRate = 1/samplePeriod;
end

% ---------------- Align Signals Once ----------------

signalLengths = zeros(numberOfModels, 1);

for modelIndex = 1:numberOfModels

    signalLengths(modelIndex) = min( ...
        numel(models(modelIndex).Reference), ...
        numel(models(modelIndex).Prediction));
end

commonLength = min(signalLengths);

if commonLength < 2
    error('The imported model outputs do not contain enough samples.');
end

samplingRate = models(1).SamplingRate;

for modelIndex = 2:numberOfModels

    relativeRateDifference = abs( ...
        models(modelIndex).SamplingRate-samplingRate) / samplingRate;

    if relativeRateDifference > 1e-3
        warning(['Model %d has a different sampling rate: ' ...
            '%.6f Hz vs. %.6f Hz.'], ...
            modelIndex, ...
            models(modelIndex).SamplingRate, ...
            samplingRate);
    end
end

referenceSignal = ...
    models(1).Reference(1:commonLength);

referenceSignal = ...
    referenceSignal-mean(referenceSignal, 'omitnan');

modelPredictions = zeros(commonLength, numberOfModels);

for modelIndex = 1:numberOfModels

    currentPrediction = ...
        models(modelIndex).Prediction(1:commonLength);

    modelPredictions(:, modelIndex) = ...
        currentPrediction-mean(currentPrediction, 'omitnan');
end

% ---------------- Create HPF Signal Once ----------------

nyquistFrequency = samplingRate/2;

if highpassCutoff <= 0 || highpassCutoff >= nyquistFrequency
    error(['The high-pass cutoff must be greater than zero and below ' ...
        'the Nyquist frequency of %.6f Hz.'], ...
        nyquistFrequency);
end

[bFilter, aFilter] = butter( ...
    filterOrder, ...
    highpassCutoff/nyquistFrequency, ...
    'high');

filteredSignal = filtfilt( ...
    bFilter, ...
    aFilter, ...
    referenceSignal);

filteredSignal = ...
    filteredSignal-mean(filteredSignal, 'omitnan');

colors = get(groot, 'defaultAxesColorOrder');

% ============================================================
%  FRF Calculation and Plot
% ============================================================

N = commonLength;
singleSidedIndices = 1:floor(N/2)+1;

frfFrequency = ...
    samplingRate*(0:floor(N/2))'/N;

referenceFFT = fft(referenceSignal);
filteredFFT = fft(filteredSignal);

referenceFFTSingle = ...
    referenceFFT(singleSidedIndices);

filteredFFTSingle = ...
    filteredFFT(singleSidedIndices);

% No denominator threshold is applied.
filteredFRF = ...
    filteredFFTSingle ./ referenceFFTSingle;

modelFRFMagnitude = zeros( ...
    numel(frfFrequency), ...
    numberOfModels);

for modelIndex = 1:numberOfModels

    modelFFT = fft(modelPredictions(:, modelIndex));

    modelFRF = ...
        modelFFT(singleSidedIndices) ./ referenceFFTSingle;

    modelFRFMagnitude(:, modelIndex) = abs(modelFRF);
end

filteredFRFMagnitude = abs(filteredFRF);

filteredFRFSmoothed = movmean( ...
    filteredFRFMagnitude, ...
    movingAverageWindow, ...
    'Endpoints', ...
    'shrink');

modelFRFSmoothed = zeros(size(modelFRFMagnitude));

for modelIndex = 1:numberOfModels

    modelFRFSmoothed(:, modelIndex) = movmean( ...
        modelFRFMagnitude(:, modelIndex), ...
        movingAverageWindow, ...
        'Endpoints', ...
        'shrink');
end

if lower(frfScale) == "db"

    filteredFRFPlot = ...
        20*log10(filteredFRFSmoothed + eps);

    modelFRFPlot = ...
        20*log10(modelFRFSmoothed + eps);

    frfYAxisLabel = 'FRF magnitude (dB)';

else

    filteredFRFPlot = filteredFRFSmoothed;
    modelFRFPlot = modelFRFSmoothed;
    frfYAxisLabel = 'amplitude ratio';
end

figFRF = figure( ...
    'Units', 'inches', ...
    'Position', [1 1 6.5 3.5], ...
    'PaperPositionMode', 'auto', ...
    'Color', 'white');

pFilteredFRF = plot( ...
    frfFrequency, ...
    filteredFRFPlot, ...
    'Color', colors(2,:), ...
    'LineWidth', 1.5);

hold on;

modelFRFHandles = gobjects(numberOfModels, 1);

for modelIndex = 1:numberOfModels

    colorIndex = mod( ...
        modelIndex+1, ...
        size(colors,1)) + 1;

    modelFRFHandles(modelIndex) = plot( ...
        frfFrequency, ...
        modelFRFPlot(:, modelIndex), ...
        'Color', colors(colorIndex,:), ...
        'LineWidth', 1.5);
end

hold off;

grid on;
box on;

xlabel('frequency (Hz)');
ylabel(frfYAxisLabel);
xlim(frequencyLimit);
ylim([0 4]);
%Optional fixed FRF limits
% if lower(frfScale) == "db"
%     ylim(frfDBLimit);
% else
%     ylim(frfLinearLimit);
% end

legend( ...
    [pFilteredFRF; modelFRFHandles], ...
    '10 Hz HPF', ...
    modelLabels{:}, ...
    'Location', ...
    'NorthEast');

set(gca, ...
    'FontName', 'Times New Roman', ...
    'FontSize', 10, ...
    'LineWidth', 0.6, ...
    'TickDir', 'out');

exportgraphics( ...
    figFRF, ...
    fullfile(outDir, 'FRF_HPF_and_Eleven_LSTM_Models.png'), ...
    'Resolution', 300, ...
    'BackgroundColor', 'white');

% ============================================================
%  PSD Calculation and Plot
% ============================================================

segmentLength = min( ...
    requestedSegmentLength, ...
    commonLength);

segmentLength = max(segmentLength, 2);

psdWindow = hann(segmentLength);

overlapLength = floor( ...
    overlapFraction*segmentLength);

nfftPSD = max( ...
    minimumNFFT, ...
    2^nextpow2(segmentLength));

[filteredPSD, psdFrequency] = pwelch( ...
    filteredSignal, ...
    psdWindow, ...
    overlapLength, ...
    nfftPSD, ...
    samplingRate);

filteredPSDdB = ...
    10*log10(filteredPSD + eps);

modelPSDdB = zeros( ...
    numel(psdFrequency), ...
    numberOfModels);

for modelIndex = 1:numberOfModels

    currentModelPSD = pwelch( ...
        modelPredictions(:, modelIndex), ...
        psdWindow, ...
        overlapLength, ...
        nfftPSD, ...
        samplingRate);

    modelPSDdB(:, modelIndex) = ...
        10*log10(currentModelPSD + eps);
end

figPSD = figure( ...
    'Units', 'inches', ...
    'Position', [1 1 6.5 3.5], ...
    'PaperPositionMode', 'auto', ...
    'Color', 'white');

pFilteredPSD = plot( ...
    psdFrequency, ...
    filteredPSDdB, ...
    'Color', colors(2,:), ...
    'LineWidth', 1.5);

hold on;

modelPSDHandles = gobjects(numberOfModels, 1);

for modelIndex = 1:numberOfModels

    colorIndex = mod( ...
        modelIndex+1, ...
        size(colors,1)) + 1;

    modelPSDHandles(modelIndex) = plot( ...
        psdFrequency, ...
        modelPSDdB(:, modelIndex), ...
        'Color', colors(colorIndex,:), ...
        'LineWidth', 1.5);
end

hold off;

grid on;
box on;

xlabel('frequency (Hz)');
ylabel('PSD (g^2/Hz, dB)');
xlim(frequencyLimit);

% Optional fixed PSD limits
% ylim(psdMagnitudeLimit);

legend( ...
    [pFilteredPSD; modelPSDHandles], ...
    '10 Hz HPF', ...
    modelLabels{:}, ...
    'Location', ...
    'southeast');

set(gca, ...
    'FontName', 'Times New Roman', ...
    'FontSize', 10, ...
    'LineWidth', 0.6, ...
    'TickDir', 'out');

exportgraphics( ...
    figPSD, ...
    fullfile(outDir, 'PSD_HPF_and_Eleven_LSTM_Models.png'), ...
    'Resolution', 300, ...
    'BackgroundColor', 'white');

fprintf('\nFRF and PSD figures saved to:\n%s\n', outDir);

%% ============================================================
%  SNR and RMSE Comparison Table
%
%  SNR improvement is calculated relative to the HPF result:
%
%  Improvement (%) =
%      100 * (SNR_model - SNR_HPF) / abs(SNR_HPF)
%
%  The absolute value in the denominator prevents the percentage sign
%  from reversing when the HPF SNR is negative.
% ============================================================

numberOfComparedSignals = numberOfModels + 1;

signalNames = strings(numberOfComparedSignals,1);
rmseValues = zeros(numberOfComparedSignals,1);
snrValuesdB = zeros(numberOfComparedSignals,1);
snrImprovementPercent = zeros(numberOfComparedSignals,1);

% ---------------- HPF Metrics ----------------

hpfError = referenceSignal - filteredSignal;

signalNames(1) = "10 Hz HPF";

rmseValues(1) = sqrt(mean( ...
    hpfError.^2, ...
    'omitnan'));

snrValuesdB(1) = 10 * log10( ...
    sum(referenceSignal.^2, 'omitnan') / ...
    sum(hpfError.^2, 'omitnan'));

% HPF is the comparison baseline.
snrImprovementPercent(1) = 0;

% ---------------- LSTM Model Metrics ----------------

for modelIndex = 1:numberOfModels

    currentPrediction = ...
        modelPredictions(:,modelIndex);

    currentError = ...
        referenceSignal - currentPrediction;

    signalNames(modelIndex+1) = ...
        string(modelLabels{modelIndex});

    rmseValues(modelIndex+1) = sqrt(mean( ...
        currentError.^2, ...
        'omitnan'));

    snrValuesdB(modelIndex+1) = 10 * log10( ...
        sum(referenceSignal.^2, 'omitnan') / ...
        sum(currentError.^2, 'omitnan'));
end

% ---------------- SNR Improvement Relative to HPF ----------------

hpfSNRdB = snrValuesdB(1);

if abs(hpfSNRdB) > eps

    snrImprovementPercent(2:end) = ...
        100 * (snrValuesdB(2:end) - hpfSNRdB) / abs(hpfSNRdB);

else

    warning(['The HPF SNR is approximately zero dB, so percentage ' ...
             'improvement cannot be calculated reliably.']);

    snrImprovementPercent(2:end) = NaN;
end

% ---------------- Create Performance Table ----------------

performanceTable = table( ...
    signalNames, ...
    rmseValues, ...
    snrValuesdB, ...
    snrImprovementPercent, ...
    'VariableNames', { ...
    'Signal', ...
    'RMSE', ...
    'SNR_dB', ...
    'SNR_Improvement_Percent'});

% ---------------- Print Table ----------------

fprintf('\n============================================================\n');
fprintf('         RMSE AND SNR PERFORMANCE COMPARISON\n');
fprintf('============================================================\n\n');

disp(performanceTable);

% ---------------- Save Table ----------------

performanceTableFile = fullfile( ...
    outDir, ...
    'RMSE_SNR_Comparison.csv');

writetable( ...
    performanceTable, ...
    performanceTableFile);

fprintf('Performance table saved to:\n%s\n', ...
    performanceTableFile);

%% ============================================================
%  SNR vs. LSTM Model Size - Logarithmic X-Axis
% ============================================================

% Model sizes
modelSizes = expectedHiddenUnits;

% snrValuesdB(1) is the HPF baseline.
% Entries 2:end correspond to the LSTM models.
modelSNRdB = snrValuesdB(2:end);

% Create figure
figSNR = figure( ...
    'Units', 'inches', ...
    'Position', [1 1 6.5 2], ...
    'PaperPositionMode', 'auto', ...
    'Color', 'white');

% Plot with logarithmic x-axis
semilogx( ...
    modelSizes, ...
    modelSNRdB, ...
    '-o', ...
    'LineWidth', 1.5, ...
    'MarkerSize', 6);

grid on;
box on;

xlabel('LSTM Hidden Units');
ylabel('SNR (dB)');

% Show the actual tested model sizes on the x-axis
xticks(modelSizes);
xticklabels(string(modelSizes));

set(gca, ...
    'XScale', 'log', ...
    'FontName', 'Times New Roman', ...
    'FontSize', 10, ...
    'LineWidth', 0.6, ...
    'TickDir', 'out');

% Export figure
snrFigureFile = fullfile( ...
    outDir, ...
    'SNR_vs_LSTM_Model_Size_LogX.png');

exportgraphics( ...
    figSNR, ...
    snrFigureFile, ...
    'Resolution', 300, ...
    'BackgroundColor', 'white');
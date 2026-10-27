%% ============================================================
%  Compress Exported LSTM Model Using Adjustable SVD Reduction
%
%  This script is based on the original CSV inference code and the supplied
%  Python/Keras SVD workflow. It performs post-training compression of the
%  pretrained LSTM without requiring the original Keras model.
%
%  Main steps:
%    1) Import WI, WH, B, FCW, FCB, NORM, and META from CSV.
%    2) Create the degraded signal exactly as in the reference script.
%    3) Run the original LSTM.
%    4) Decompose the recurrent weights using SVD.
%    5) Optionally shrink small singular values, similar in purpose to the
%       regularized singular-value model in the Python reference.
%    6) Retain an adjustable number of singular components.
%    7) Run the compressed model directly from the reduced factors.
%    8) Compare model size, theoretical computation, timing, and error.
%    9) Export the compressed matrices for later Teensy implementation.
%
%  IMPORTANT DIFFERENCE FROM THE PYTHON CODE
%  -----------------------------------------
%  The Python code creates a trainable singular-value Keras model and then
%  fine-tunes it using training data. This MATLAB script starts with a
%  post-training SVD approximation of the exported weights. It can perform
%  singular-value shrinkage, but it does not retrain the factors unless a
%  separate training stage is later added.
%
%  WHY WH IS THE MAIN COMPRESSION TARGET
%  -------------------------------------
%  For this model:
%       WI  is 4H-by-1 and has rank at most 1.
%       WH  is 4H-by-H and contains most model parameters.
%       FCW is 1-by-H and has rank at most 1.
%
%  Therefore, useful SVD compression is concentrated in WH.
%% ============================================================

clc;
clear;
close all;

%% ---------------- User Settings ----------------

modelFolder = ['C:\Users\joud\Dropbox\' ...
    'Satme_2026_IEEE_Edge_Processing_Compensator\Code\' ...
    'MATLAB\Model 5\Teensy_LSTM_Parameters'];

dataFolder = ['C:\Users\joud\Dropbox\' ...
    'Satme_2026_IEEE_Edge_Processing_Compensator\Code\MATLAB'];

rawFile = fullfile(dataFolder, 'RAW.CSV');

% Time-unit options:
%   "auto", "seconds", "milliseconds", or "microseconds"
timeUnit = "auto";

% Degradation filter settings
filterOrder = 5;
highpassCutoff = 10;  % Hz

%% ---------------- Compression Strategy ----------------
%
% "wholeMatrix"
%   Applies one SVD to the complete 4H-by-H recurrent matrix.
%
% "perGate"
%   Separates the input, forget, candidate, and output recurrent matrices,
%   then compresses each H-by-H matrix independently.
%
% Per-gate compression gives more control because each LSTM gate may require
% a different retained rank. This is closest to the idea in the supplied
% Python work that different LSTM matrices can be reduced independently.
compressionStrategy = "perGate";

%% ---------------- Rank Selection ----------------
%
% "fixedRank"
%   Uses ranks specified below.
%
% "energy"
%   Selects the smallest rank that retains the requested fraction of
%   squared singular-value energy.
rankSelectionMode = "energy";

% Used by wholeMatrix compression with fixedRank.
wholeMatrixFixedRank = 60;

% Used by perGate compression with fixedRank.
% Order: [input gate, forget gate, candidate gate, output gate]
gateFixedRanks = [40 40 40 40];

% Used when rankSelectionMode = "energy".
% Can be one value for all gates or four separate values.
wholeMatrixEnergyTarget = 0.999;
gateEnergyTargets = [0.99 0.99 0.99 0.99];

% Optional maximum ranks. Use Inf to disable.
wholeMatrixMaximumRank = Inf;
gateMaximumRanks = [Inf Inf Inf Inf];

%% ---------------- Singular-Value Regularization ----------------
%
% The supplied Python code rebuilds the network with trainable singular
% factors and applies regularization before fine-tuning. Without retraining,
% the closest post-training analogue is singular-value shrinkage.
%
% "none"
%   Keep retained singular values unchanged.
%
% "softThreshold"
%   Replace each singular value s with max(s-lambda,0).
%
% "relativeThreshold"
%   Remove singular values below threshold*largest singular value.
%
% Start with "none". Shrinkage is more aggressive and can increase error.
singularValueRegularization = "none";

% Used only for "softThreshold".
% This is an absolute value in the same units as the singular values.
softThresholdLambda = 0;

% Used only for "relativeThreshold".
% Example: 0.01 removes singular values below 1 percent of the largest.
relativeSingularThreshold = 0.001;

%% ---------------- Output Settings ----------------

saveCompressedModel = true;
saveComparisonOutput = true;

compressedModelFolder = fullfile( ...
    modelFolder, ...
    'SVD_Compressed_Model_Adjustable');

comparisonOutputFile = fullfile( ...
    compressedModelFolder, ...
    'MATLAB_SVD_COMPARISON.CSV');

%% ---------------- Load Metadata ----------------

meta = readmatrix(fullfile(modelFolder, 'META.CSV'));
meta = meta(isfinite(meta));

if isempty(meta)
    error('META.CSV is empty or could not be read.');
end

hiddenSize = round(meta(1));

if hiddenSize <= 0
    error('Invalid hidden-unit count in META.CSV: %g', meta(1));
end

fprintf('Hidden units: %d\n', hiddenSize);

%% ---------------- Load Exported Parameters ----------------

Wi  = readmatrix(fullfile(modelFolder, 'WI.CSV'));
Wh  = readmatrix(fullfile(modelFolder, 'WH.CSV'));
b   = readmatrix(fullfile(modelFolder, 'B.CSV'));
Wfc = readmatrix(fullfile(modelFolder, 'FCW.CSV'));
bfc = readmatrix(fullfile(modelFolder, 'FCB.CSV'));
normValues = readmatrix(fullfile(modelFolder, 'NORM.CSV'));

Wi = Wi(isfinite(Wi));
b = b(isfinite(b));
Wfc = Wfc(isfinite(Wfc));
bfc = bfc(isfinite(bfc));
normValues = normValues(isfinite(normValues));

Wh = Wh(:, any(isfinite(Wh),1));
Wh = Wh(any(isfinite(Wh),2),:);

Wi = Wi(:);
b = b(:);
Wfc = Wfc(:).';

if isempty(bfc)
    error('FCB.CSV is empty or could not be read.');
end

bfc = bfc(1);

%% ---------------- Validate Dimensions ----------------

expectedGateRows = 4 * hiddenSize;

if numel(Wi) ~= expectedGateRows
    error('WI.CSV contains %d values; expected %d.', ...
        numel(Wi), expectedGateRows);
end

if ~isequal(size(Wh), [expectedGateRows, hiddenSize])
    error('WH.CSV has size %d-by-%d; expected %d-by-%d.', ...
        size(Wh,1), size(Wh,2), expectedGateRows, hiddenSize);
end

if numel(b) ~= expectedGateRows
    error('B.CSV contains %d values; expected %d.', ...
        numel(b), expectedGateRows);
end

if numel(Wfc) ~= hiddenSize
    error('FCW.CSV contains %d values; expected %d.', ...
        numel(Wfc), hiddenSize);
end

if numel(normValues) < 4
    error('NORM.CSV must contain at least four normalization values.');
end

inputMean = normValues(1);
inputStd = normValues(2);
targetMean = normValues(3);
targetStd = normValues(4);

if inputStd == 0 || targetStd == 0
    error('A normalization standard deviation is zero.');
end

fprintf('Exported model dimensions are valid.\n');

%% ---------------- Load Input Data ----------------

if ~isfile(rawFile)
    error('Input file was not found: %s', rawFile);
end

rawData = readmatrix(rawFile);
rawData = rawData(:, any(isfinite(rawData),1));
rawData = rawData(any(isfinite(rawData),2),:);

if size(rawData,2) < 2
    error('RAW.CSV must contain at least time and signal columns.');
end

timeRaw = rawData(:,1);
rawSignal = rawData(:,2);

hasIndependentReference = size(rawData,2) >= 3;

if hasIndependentReference
    referenceSignal = rawData(:,3);
    validRows = isfinite(timeRaw) & ...
                isfinite(rawSignal) & ...
                isfinite(referenceSignal);
else
    validRows = isfinite(timeRaw) & isfinite(rawSignal);
end

timeRaw = timeRaw(validRows);
rawSignal = rawSignal(validRows);

if hasIndependentReference
    referenceSignal = referenceSignal(validRows);
else
    referenceSignal = rawSignal;
end

rawSignal = rawSignal - mean(rawSignal,'omitnan');
referenceSignal = referenceSignal - mean(referenceSignal,'omitnan');

%% ---------------- Convert Time and Create Degraded Input ----------------

timeSeconds = convertTimeToSeconds(timeRaw,timeUnit);
timeSeconds = timeSeconds - timeSeconds(1);

samplePeriod = mean(diff(timeSeconds),'omitnan');

if ~isfinite(samplePeriod) || samplePeriod <= 0
    error('The calculated sampling period is invalid.');
end

samplingRate = 1 / samplePeriod;
nyquistFrequency = samplingRate / 2;

if highpassCutoff <= 0 || highpassCutoff >= nyquistFrequency
    error('The high-pass cutoff must be below %.6f Hz.', ...
        nyquistFrequency);
end

[filterNumerator,filterDenominator] = butter( ...
    filterOrder, ...
    highpassCutoff/nyquistFrequency, ...
    'high');

degradedSignal = filtfilt( ...
    filterNumerator, ...
    filterDenominator, ...
    rawSignal);

degradedSignal = degradedSignal - mean(degradedSignal,'omitnan');

fprintf('Samples loaded: %d\n',numel(degradedSignal));
fprintf('Estimated sampling rate: %.6f samples/s\n',samplingRate);

%% ---------------- Build Compressed Recurrent Model ----------------

gateNames = ["Input","Forget","Candidate","Output"];

switch lower(compressionStrategy)

    case "wholematrix"

        [leftFactors,rightFactors,compressionInfo] = ...
            compressMatrixWithSVD( ...
                Wh, ...
                rankSelectionMode, ...
                wholeMatrixFixedRank, ...
                wholeMatrixEnergyTarget, ...
                wholeMatrixMaximumRank, ...
                singularValueRegularization, ...
                softThresholdLambda, ...
                relativeSingularThreshold);

        compressedModel.type = "wholeMatrix";
        compressedModel.left = leftFactors;
        compressedModel.right = rightFactors;

        fprintf('\nWhole recurrent matrix compression\n');
        printCompressionInfo("WH",compressionInfo);

    case "pergate"

        compressedModel.type = "perGate";
        compressedModel.left = cell(4,1);
        compressedModel.right = cell(4,1);
        % Preallocate the structure with the same fields returned by
        % compressMatrixWithSVD. MATLAB cannot assign a populated structure
        % into an empty structure array with no defined fields.
        emptyCompressionInfo = struct( ...
            'retainedRank', [], ...
            'retainedEnergy', [], ...
            'relativeError', [], ...
            'singularValues', [], ...
            'adjustedSingularValues', []);

        compressionInfo = repmat(emptyCompressionInfo,4,1);

        for gateIndex = 1:4

            rowRange = ...
                (gateIndex-1)*hiddenSize + (1:hiddenSize);

            gateMatrix = Wh(rowRange,:);

            [compressedModel.left{gateIndex}, ...
             compressedModel.right{gateIndex}, ...
             compressionInfo(gateIndex)] = ...
                compressMatrixWithSVD( ...
                    gateMatrix, ...
                    rankSelectionMode, ...
                    gateFixedRanks(gateIndex), ...
                    gateEnergyTargets(gateIndex), ...
                    gateMaximumRanks(gateIndex), ...
                    singularValueRegularization, ...
                    softThresholdLambda, ...
                    relativeSingularThreshold);

            printCompressionInfo( ...
                gateNames(gateIndex) + " gate", ...
                compressionInfo(gateIndex));
        end

    otherwise

        error('Use "wholeMatrix" or "perGate".');
end

%% ---------------- Parameter and MAC Counts ----------------

originalWhParameters = numel(Wh);
originalRecurrentMACs = numel(Wh);

if compressedModel.type == "wholeMatrix"

    compressedWhParameters = ...
        numel(compressedModel.left) + ...
        numel(compressedModel.right);

    compressedRecurrentMACs = compressedWhParameters;

else

    compressedWhParameters = 0;
    compressedRecurrentMACs = 0;

    for gateIndex = 1:4
        compressedWhParameters = compressedWhParameters + ...
            numel(compressedModel.left{gateIndex}) + ...
            numel(compressedModel.right{gateIndex});

        compressedRecurrentMACs = compressedRecurrentMACs + ...
            numel(compressedModel.left{gateIndex}) + ...
            numel(compressedModel.right{gateIndex});
    end
end

unchangedParameters = ...
    numel(Wi) + numel(b) + numel(Wfc) + numel(bfc);

originalTotalParameters = ...
    unchangedParameters + originalWhParameters;

compressedTotalParameters = ...
    unchangedParameters + compressedWhParameters;

whReductionPercent = 100 * ...
    (1-compressedWhParameters/originalWhParameters);

totalReductionPercent = 100 * ...
    (1-compressedTotalParameters/originalTotalParameters);

macReductionPercent = 100 * ...
    (1-compressedRecurrentMACs/originalRecurrentMACs);

fprintf('\n----- Compression summary -----\n');
fprintf('Original WH parameters:       %d\n',originalWhParameters);
fprintf('Compressed WH parameters:     %d\n',compressedWhParameters);
fprintf('WH parameter reduction:       %.3f %%\n',whReductionPercent);

fprintf('Original total parameters:    %d\n',originalTotalParameters);
fprintf('Compressed total parameters:  %d\n',compressedTotalParameters);
fprintf('Total parameter reduction:    %.3f %%\n',totalReductionPercent);

fprintf('Original recurrent MACs:      %d per sample\n', ...
    originalRecurrentMACs);
fprintf('Compressed recurrent MACs:    %d per sample\n', ...
    compressedRecurrentMACs);
fprintf('Theoretical MAC reduction:    %.3f %%\n', ...
    macReductionPercent);

fprintf('Estimated original float size:   %.3f kB\n', ...
    4*originalTotalParameters/1024);
fprintf('Estimated compressed float size: %.3f kB\n', ...
    4*compressedTotalParameters/1024);

if compressedWhParameters >= originalWhParameters
    warning(['The selected rank does not reduce recurrent parameter ' ...
        'storage. Select a smaller rank or lower energy target.']);
end

%% ---------------- Run Original Model ----------------

fprintf('\nRunning original model...\n');
originalTimer = tic;

originalCompensatedSignal = runOriginalLSTM( ...
    degradedSignal,Wi,Wh,b,Wfc,bfc, ...
    inputMean,inputStd,targetMean,targetStd,hiddenSize);

originalElapsedTime = toc(originalTimer);

%% ---------------- Run Compressed Model ----------------

fprintf('Running compressed model...\n');
compressedTimer = tic;

compressedCompensatedSignal = runCompressedLSTM( ...
    degradedSignal,Wi,compressedModel,b,Wfc,bfc, ...
    inputMean,inputStd,targetMean,targetStd,hiddenSize);

compressedElapsedTime = toc(compressedTimer);

numberOfSamples = numel(degradedSignal);

fprintf('\n----- Timing comparison -----\n');
fprintf('Original inference:   %.3f s, %.3f us/sample\n', ...
    originalElapsedTime, ...
    1e6*originalElapsedTime/numberOfSamples);

fprintf('Compressed inference: %.3f s, %.3f us/sample\n', ...
    compressedElapsedTime, ...
    1e6*compressedElapsedTime/numberOfSamples);

fprintf('Measured MATLAB speed ratio: %.3f x\n', ...
    originalElapsedTime/compressedElapsedTime);

fprintf(['Desktop MATLAB timing may differ from Teensy timing because ' ...
    'small matrix operations have different overhead.\n']);

%% ---------------- Accuracy Metrics ----------------

originalError = originalCompensatedSignal-referenceSignal;
compressedError = compressedCompensatedSignal-referenceSignal;
compressionOnlyError = ...
    compressedCompensatedSignal-originalCompensatedSignal;

originalRMSE = sqrt(mean(originalError.^2,'omitnan'));
compressedRMSE = sqrt(mean(compressedError.^2,'omitnan'));
compressionRMSE = sqrt(mean(compressionOnlyError.^2,'omitnan'));

originalMAE = mean(abs(originalError),'omitnan');
compressedMAE = mean(abs(compressedError),'omitnan');
compressionMAE = mean(abs(compressionOnlyError),'omitnan');

fprintf('\n----- Accuracy comparison -----\n');
fprintf('Original vs reference RMSE:   %.10g\n',originalRMSE);
fprintf('Compressed vs reference RMSE: %.10g\n',compressedRMSE);
fprintf('Compressed vs original RMSE:  %.10g\n',compressionRMSE);

fprintf('Original vs reference MAE:    %.10g\n',originalMAE);
fprintf('Compressed vs reference MAE:  %.10g\n',compressedMAE);
fprintf('Compressed vs original MAE:   %.10g\n',compressionMAE);

%% ---------------- Plot Singular Values ----------------

figure( ...
    'Units','inches', ...
    'Position',[1 1 6.5 3.8], ...
    'Color','white');

if compressedModel.type == "wholeMatrix"

    semilogy( ...
        compressionInfo.singularValues, ...
        'LineWidth',1.5);

    hold on;
    xline(compressionInfo.retainedRank,'--', ...
        sprintf('rank = %d',compressionInfo.retainedRank));
    hold off;

    legend('WH singular values','Location','northeast');

else

    hold on;

    for gateIndex = 1:4
        semilogy( ...
            compressionInfo(gateIndex).singularValues, ...
            'LineWidth',1.2);
    end

    hold off;

    legend( ...
        'input gate', ...
        'forget gate', ...
        'candidate gate', ...
        'output gate', ...
        'Location','northeast');
end

grid on;
box on;
xlabel('singular-value index');
ylabel('singular value');

%% ---------------- Plot Model Outputs ----------------

colors = get(groot,'defaultAxesColorOrder');

figure( ...
    'Units','inches', ...
    'Position',[1 1 6.5 4.4], ...
    'Color','white');

layout = tiledlayout(2,1, ...
    'TileSpacing','compact', ...
    'Padding','compact');

nexttile;

plot(timeSeconds,referenceSignal, ...
    'Color',colors(1,:), ...
    'LineWidth',0.2);

hold on;

plot(timeSeconds,originalCompensatedSignal, ...
    'Color',colors(2,:), ...
    'LineWidth',0.2);

plot(timeSeconds,compressedCompensatedSignal, ...
    'Color',colors(3,:), ...
    'LineWidth',0.2);

hold off;

grid on;
box on;
xlabel('time (s)');
ylabel('acceleration (g)');
xlim([0 max(timeSeconds)]);

legend( ...
    'reference signal', ...
    'original LSTM', ...
    'compressed LSTM', ...
    'Location','northwest');

nexttile;

plot(timeSeconds,compressionOnlyError,'LineWidth',0.5);

grid on;
box on;
xlabel('time (s)');
ylabel('compressed - original (g)');
xlim([0 max(timeSeconds)]);

%% ---------------- Save Compressed Parameters ----------------

if saveCompressedModel

    if ~exist(compressedModelFolder,'dir')
        mkdir(compressedModelFolder);
    end

    writematrix(Wi,fullfile(compressedModelFolder,'WI.CSV'));
    writematrix(b,fullfile(compressedModelFolder,'B.CSV'));
    writematrix(Wfc,fullfile(compressedModelFolder,'FCW.CSV'));
    writematrix(bfc,fullfile(compressedModelFolder,'FCB.CSV'));
    writematrix(normValues,fullfile(compressedModelFolder,'NORM.CSV'));

    if compressedModel.type == "wholeMatrix"

        writematrix( ...
            compressedModel.left, ...
            fullfile(compressedModelFolder,'WH_LEFT.CSV'));

        writematrix( ...
            compressedModel.right, ...
            fullfile(compressedModelFolder,'WH_RIGHT.CSV'));

        retainedRanks = compressionInfo.retainedRank;
        retainedEnergies = compressionInfo.retainedEnergy;
        reconstructionErrors = compressionInfo.relativeError;

    else

        filePrefixes = ["I","F","C","O"];
        retainedRanks = zeros(4,1);
        retainedEnergies = zeros(4,1);
        reconstructionErrors = zeros(4,1);

        for gateIndex = 1:4

            writematrix( ...
                compressedModel.left{gateIndex}, ...
                fullfile( ...
                    compressedModelFolder, ...
                    "WH_" + filePrefixes(gateIndex) + "_LEFT.CSV"));

            writematrix( ...
                compressedModel.right{gateIndex}, ...
                fullfile( ...
                    compressedModelFolder, ...
                    "WH_" + filePrefixes(gateIndex) + "_RIGHT.CSV"));

            retainedRanks(gateIndex) = ...
                compressionInfo(gateIndex).retainedRank;

            retainedEnergies(gateIndex) = ...
                compressionInfo(gateIndex).retainedEnergy;

            reconstructionErrors(gateIndex) = ...
                compressionInfo(gateIndex).relativeError;
        end
    end

    compressionMetadata = [ ...
        hiddenSize;
        retainedRanks(:);
        retainedEnergies(:);
        reconstructionErrors(:);
        originalTotalParameters;
        compressedTotalParameters;
        originalRecurrentMACs;
        compressedRecurrentMACs];

    writematrix( ...
        compressionMetadata, ...
        fullfile(compressedModelFolder,'META_SVD.CSV'));

    fprintf('\nCompressed model saved to:\n%s\n', ...
        compressedModelFolder);
end

%% ---------------- Save Signal Comparison ----------------

if saveComparisonOutput

    if ~exist(compressedModelFolder,'dir')
        mkdir(compressedModelFolder);
    end

    comparisonTable = table( ...
        timeRaw, ...
        referenceSignal, ...
        degradedSignal, ...
        originalCompensatedSignal, ...
        compressedCompensatedSignal, ...
        compressionOnlyError, ...
        'VariableNames',{ ...
        'Time', ...
        'Reference', ...
        'Degraded', ...
        'OriginalCompensated', ...
        'CompressedCompensated', ...
        'CompressionError'});

    writetable(comparisonTable,comparisonOutputFile);
end

%% ============================================================
%  Local Functions
% ============================================================

function [leftFactor,rightFactor,info] = compressMatrixWithSVD( ...
    matrix, ...
    rankSelectionMode, ...
    fixedRank, ...
    energyTarget, ...
    maximumRank, ...
    regularizationMode, ...
    softThresholdLambda, ...
    relativeThreshold)

    [U,S,V] = svd(matrix,'econ');

    originalSingularValues = diag(S);
    adjustedSingularValues = originalSingularValues;

    switch lower(regularizationMode)

        case "none"

            % No modification.

        case "softthreshold"

            adjustedSingularValues = max( ...
                originalSingularValues-softThresholdLambda, ...
                0);

        case "relativethreshold"

            thresholdValue = ...
                relativeThreshold*originalSingularValues(1);

            adjustedSingularValues( ...
                adjustedSingularValues<thresholdValue) = 0;

        otherwise

            error(['Use "none", "softThreshold", or ' ...
                '"relativeThreshold".']);
    end

    nonzeroCount = nnz(adjustedSingularValues>0);

    if nonzeroCount == 0
        error('Regularization removed every singular value.');
    end

    energy = adjustedSingularValues.^2;
    cumulativeEnergy = cumsum(energy)/sum(energy);

    switch lower(rankSelectionMode)

        case "fixedrank"

            retainedRank = round(fixedRank);

        case "energy"

            retainedRank = find( ...
                cumulativeEnergy>=energyTarget, ...
                1, ...
                'first');

        otherwise

            error('Use "fixedRank" or "energy".');
    end

    if isempty(retainedRank)
        retainedRank = nonzeroCount;
    end

    if isfinite(maximumRank)
        retainedRank = min(retainedRank,round(maximumRank));
    end

    retainedRank = max(1,min( ...
        [retainedRank,nonzeroCount,size(S,1)]));

    retainedValues = ...
        adjustedSingularValues(1:retainedRank);

    Ur = U(:,1:retainedRank);
    Vr = V(:,1:retainedRank);

    % Split each singular value evenly between the two factors:
    %
    %   U*S*V' =
    %   (U*sqrt(S)) * (sqrt(S)*V')
    %
    % This balanced form resembles a two-layer factorized implementation.
    squareRootValues = sqrt(retainedValues);

    leftFactor = Ur .* squareRootValues.';
    rightFactor = squareRootValues .* Vr.';

    approximation = leftFactor*rightFactor;

    info.retainedRank = retainedRank;
    info.retainedEnergy = cumulativeEnergy(retainedRank);
    info.relativeError = ...
        norm(matrix-approximation,'fro')/norm(matrix,'fro');

    info.singularValues = originalSingularValues;
    info.adjustedSingularValues = adjustedSingularValues;
end

function printCompressionInfo(matrixName,info)

    fprintf('\n%s compression:\n',matrixName);
    fprintf('  Retained rank:             %d\n', ...
        info.retainedRank);
    fprintf('  Retained adjusted energy:  %.8f %%\n', ...
        100*info.retainedEnergy);
    fprintf('  Relative matrix error:     %.10g\n', ...
        info.relativeError);
end

function outputSignal = runOriginalLSTM( ...
    inputSignal,Wi,Wh,b,Wfc,bfc, ...
    inputMean,inputStd,targetMean,targetStd,hiddenSize)

    sampleCount = numel(inputSignal);

    hiddenState = zeros(hiddenSize,1);
    cellState = zeros(hiddenSize,1);
    outputSignal = zeros(sampleCount,1);

    idxInput = 1:hiddenSize;
    idxForget = hiddenSize+(1:hiddenSize);
    idxCandidate = 2*hiddenSize+(1:hiddenSize);
    idxOutput = 3*hiddenSize+(1:hiddenSize);

    for sampleIndex = 1:sampleCount

        normalizedInput = ...
            (inputSignal(sampleIndex)-inputMean)/inputStd;

        gateVector = ...
            Wi*normalizedInput + ...
            Wh*hiddenState + ...
            b;

        inputGate = ...
            logisticSigmoid(gateVector(idxInput));

        forgetGate = ...
            logisticSigmoid(gateVector(idxForget));

        cellCandidate = ...
            tanh(gateVector(idxCandidate));

        outputGate = ...
            logisticSigmoid(gateVector(idxOutput));

        cellState = ...
            forgetGate.*cellState + ...
            inputGate.*cellCandidate;

        hiddenState = ...
            outputGate.*tanh(cellState);

        normalizedOutput = Wfc*hiddenState+bfc;

        outputSignal(sampleIndex) = ...
            normalizedOutput*targetStd+targetMean;
    end
end

function outputSignal = runCompressedLSTM( ...
    inputSignal,Wi,compressedModel,b,Wfc,bfc, ...
    inputMean,inputStd,targetMean,targetStd,hiddenSize)

    sampleCount = numel(inputSignal);

    hiddenState = zeros(hiddenSize,1);
    cellState = zeros(hiddenSize,1);
    outputSignal = zeros(sampleCount,1);

    idxInput = 1:hiddenSize;
    idxForget = hiddenSize+(1:hiddenSize);
    idxCandidate = 2*hiddenSize+(1:hiddenSize);
    idxOutput = 3*hiddenSize+(1:hiddenSize);

    for sampleIndex = 1:sampleCount

        normalizedInput = ...
            (inputSignal(sampleIndex)-inputMean)/inputStd;

        if compressedModel.type == "wholeMatrix"

            recurrentTerm = ...
                compressedModel.left * ...
                (compressedModel.right*hiddenState);

        else

            recurrentTerm = zeros(4*hiddenSize,1);

            for gateIndex = 1:4

                rowRange = ...
                    (gateIndex-1)*hiddenSize + ...
                    (1:hiddenSize);

                recurrentTerm(rowRange) = ...
                    compressedModel.left{gateIndex} * ...
                    (compressedModel.right{gateIndex}*hiddenState);
            end
        end

        gateVector = ...
            Wi*normalizedInput + ...
            recurrentTerm + ...
            b;

        inputGate = ...
            logisticSigmoid(gateVector(idxInput));

        forgetGate = ...
            logisticSigmoid(gateVector(idxForget));

        cellCandidate = ...
            tanh(gateVector(idxCandidate));

        outputGate = ...
            logisticSigmoid(gateVector(idxOutput));

        cellState = ...
            forgetGate.*cellState + ...
            inputGate.*cellCandidate;

        hiddenState = ...
            outputGate.*tanh(cellState);

        normalizedOutput = Wfc*hiddenState+bfc;

        outputSignal(sampleIndex) = ...
            normalizedOutput*targetStd+targetMean;
    end
end

function y = logisticSigmoid(x)

    y = zeros(size(x));

    positive = x>=0;
    negative = ~positive;

    y(positive) = 1./(1+exp(-x(positive)));

    expX = exp(x(negative));
    y(negative) = expX./(1+expX);
end

function timeSeconds = convertTimeToSeconds(timeRaw,timeUnit)

    timeUnit = lower(string(timeUnit));

    if timeUnit == "seconds"

        scaleFactor = 1;

    elseif timeUnit == "milliseconds"

        scaleFactor = 1e-3;

    elseif timeUnit == "microseconds"

        scaleFactor = 1e-6;

    elseif timeUnit == "auto"

        medianDifference = median(diff(timeRaw),'omitnan');

        if ~isfinite(medianDifference) || medianDifference<=0
            error('The time column does not contain increasing values.');
        end

        candidateScales = [1,1e-3,1e-6];
        candidateRates = ...
            1./(medianDifference.*candidateScales);

        expectedRate = 400;

        [~,bestIndex] = min( ...
            abs(log10(candidateRates/expectedRate)));

        scaleFactor = candidateScales(bestIndex);

        unitNames = [ ...
            "seconds", ...
            "milliseconds", ...
            "microseconds"];

        fprintf('Automatically interpreted time as %s.\n', ...
            unitNames(bestIndex));

    else

        error(['Use "auto", "seconds", "milliseconds", ' ...
            'or "microseconds".']);
    end

    timeSeconds = timeRaw*scaleFactor;
end


%% ---------------- FRF: Model Before and After Compression ----------------

N = numberOfSamples;

% Single-sided frequency vector
frequency = samplingRate * (0:floor(N/2))' / N;

% Calculate FFTs
referenceFFT  = fft(referenceSignal);
originalFFT   = fft(originalCompensatedSignal);
compressedFFT = fft(compressedCompensatedSignal);

% Use only the single-sided FFT range
singleSidedIndices = 1:floor(N/2)+1;

frequencySingle = frequency;

referenceFFTSingle = referenceFFT(singleSidedIndices);
originalFFTSingle = originalFFT(singleSidedIndices);
compressedFFTSingle = compressedFFT(singleSidedIndices);

% Ignore frequency bins where the reference FFT magnitude is extremely
% small. This prevents unstable FRF values caused by division by values
% near zero.
referenceThreshold = ...
    1e-10 * max(abs(referenceFFTSingle));

validBins = ...
    abs(referenceFFTSingle) >= referenceThreshold;

% Initialize FRFs with NaN so invalid bins are not plotted.
FRF_Original = nan(size(referenceFFTSingle));
FRF_Compressed = nan(size(referenceFFTSingle));

% FRF = model output FFT / reference FFT
FRF_Original(validBins) = ...
    originalFFTSingle(validBins) ./ ...
    referenceFFTSingle(validBins);

FRF_Compressed(validBins) = ...
    compressedFFTSingle(validBins) ./ ...
    referenceFFTSingle(validBins);

% Convert FRF magnitude to decibels.
FRF_OriginalMagnitudeDB = ...
    20 * log10(abs(FRF_Original));

FRF_CompressedMagnitudeDB = ...
    20 * log10(abs(FRF_Compressed));

%% ---------------- Plot FRF Comparison ----------------

colors = get(groot,'defaultAxesColorOrder');

figure( ...
    'Units','inches', ...
    'Position',[1 1 6.5 3.2], ...
    'PaperPositionMode','auto', ...
    'Color','white');

plot( ...
    frequencySingle, ...
    FRF_OriginalMagnitudeDB, ...
    'Color',colors(2,:), ...
    'LineWidth',1.0);

hold on;

plot( ...
    frequencySingle, ...
    FRF_CompressedMagnitudeDB, ...
    'Color',colors(3,:), ...
    'LineWidth',1.0);

hold off;

grid on;
box on;

xlabel('frequency (Hz)');
ylabel('FRF magnitude (dB)');

legend( ...
    'original LSTM', ...
    'compressed LSTM', ...
    'Location','northwest');

% Adjustable frequency range
xlim([0 25]);

% Optional adjustable FRF magnitude range
% ylim([-20 20]);

set(gca, ...
    'FontName','Times New Roman', ...
    'FontSize',10, ...
    'LineWidth',0.6, ...
    'TickDir','out');


%% ================= Performance Comparison =================

% Model sizes (assuming single-precision floating point, 4 bytes/value)
originalModelSizeKB   = originalTotalParameters   * 4 / 1024;
compressedModelSizeKB = compressedTotalParameters * 4 / 1024;

% Average execution time per sample
originalTimePerSampleUS   = originalElapsedTime   / numberOfSamples * 1e6;
compressedTimePerSampleUS = compressedElapsedTime / numberOfSamples * 1e6;

% Percentage improvements
parameterReduction = ...
    100 * (1 - compressedTotalParameters / originalTotalParameters);

memoryReduction = ...
    100 * (1 - compressedModelSizeKB / originalModelSizeKB);

macReduction = ...
    100 * (1 - compressedRecurrentMACs / originalRecurrentMACs);

speedup = ...
    originalElapsedTime / compressedElapsedTime;

% ================= Performance Comparison =================

% Model sizes (assuming single-precision floating point, 4 bytes/value)
originalModelSizeKB   = originalTotalParameters   * 4 / 1024;
compressedModelSizeKB = compressedTotalParameters * 4 / 1024;

% Average execution time per sample
originalTimePerSampleUS   = originalElapsedTime   / numberOfSamples * 1e6;
compressedTimePerSampleUS = compressedElapsedTime / numberOfSamples * 1e6;

% Percentage improvements
parameterReduction = ...
    100 * (1 - compressedTotalParameters / originalTotalParameters);

memoryReduction = ...
    100 * (1 - compressedModelSizeKB / originalModelSizeKB);

macReduction = ...
    100 * (1 - compressedRecurrentMACs / originalRecurrentMACs);

speedup = ...
    originalElapsedTime / compressedElapsedTime;

% Create comparison table

PerformanceTable = table( ...
    ["Original LSTM"; "Compressed LSTM"], ...
    [originalTotalParameters; compressedTotalParameters], ...
    [originalModelSizeKB; compressedModelSizeKB], ...
    [originalRecurrentMACs; compressedRecurrentMACs], ...
    [originalElapsedTime; compressedElapsedTime], ...
    [originalTimePerSampleUS; compressedTimePerSampleUS], ...
    [originalRMSE; compressedRMSE], ...
    [originalMAE; compressedMAE], ...
    'VariableNames', ...
    {'Model', ...
     'Parameters', ...
     'ModelSize_kB', ...
     'Recurrent_MACs', ...
     'InferenceTime_s', ...
     'TimePerSample_us', ...
     'RMSE', ...
     'MAE'});

% Print table

fprintf('\n==============================================================\n');
fprintf('           LSTM Compression Performance Comparison\n');
fprintf('==============================================================\n\n');

disp(PerformanceTable)

fprintf('--------------------------------------------------------------\n');
fprintf('Parameter Reduction : %8.2f %%\n', parameterReduction);
fprintf('Memory Reduction    : %8.2f %%\n', memoryReduction);
fprintf('MAC Reduction       : %8.2f %%\n', macReduction);
fprintf('Measured Speedup    : %8.2f x\n', speedup);
fprintf('--------------------------------------------------------------\n');
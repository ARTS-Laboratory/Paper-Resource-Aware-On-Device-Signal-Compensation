/*
  ======================================================================
  LSTM_6Unit.ino
  Teensy 4.0 - 6-Unit LSTM Standalone Stage-Power Benchmark
  ======================================================================

  Purpose
  -------
  Benchmarks the 6-hidden-unit LSTM using real input data from RAW.CSV
  and generates exactly SIX 300 ms marker pulses on pin 2.

  Stage pulses:
    1) SD card initialization
    2) Model-weight loading
    3) RAW.CSV loading
    4) Warm-up forward passes
    5) Timed repeated forward passes
    6) BM6.CSV result writing

  Trigger behavior
  ----------------
  Pin 2 starts LOW.
  Each stage begins with one 300 ms HIGH pulse.
  Pin 2 returns LOW after each pulse and remains LOW during the stage.

  LED behavior
  ------------
  LED turns ON at the start of setup().
  LED stays ON through the entire benchmark cycle.
  LED turns OFF only after BM6.CSV is successfully written,
  or immediately if an error occurs.

  IMPORTANT
  ---------
  Serial diagnostics are allowed before and after the timed benchmark,
  but there are NO Serial calls and NO SD I/O inside the timed loop.

  Expected SD-card files
  ----------------------
      RAW.CSV
      W0.CSV
      W1.CSV
      B.CSV
      FCW.CSV
      FCB.CSV

  Output file
  -----------
      BM6.CSV

  ======================================================================
*/

#include <Arduino.h>
#include <SD.h>
#include <SPI.h>
#include <math.h>

// ---------------------------------------------------------------------
// User Settings
// ---------------------------------------------------------------------

constexpr uint8_t SD_CS_PIN  = 4;
constexpr uint8_t LED_PIN    = 3;
constexpr uint8_t TRIGGER_PIN = 2;

constexpr uint32_t STAGE_PULSE_MS = 300;

// Model architecture
constexpr int INPUT_SIZE  = 1;
constexpr int HIDDEN_SIZE = 6;
constexpr int GATE_SIZE   = 4 * HIDDEN_SIZE;
constexpr int WINDOW_SIZE = 400;

// RAW.CSV starting row
constexpr size_t RAW_START_SAMPLE = 0;

// Benchmark settings
constexpr int WARMUP_RUNS    = 10;
constexpr int BENCHMARK_RUNS = 1000;

// SD filenames
const char *FILE_W0 = "W0.CSV";
const char *FILE_W1 = "W1.CSV";
const char *FILE_B  = "B.CSV";
const char *FILE_FC = "FCW.CSV";
const char *FILE_FB = "FCB.CSV";

const char *RAW_FILE    = "RAW.CSV";
const char *RESULT_FILE = "BM6.CSV";

// ---------------------------------------------------------------------
// Model / Runtime Memory
// ---------------------------------------------------------------------

float *inputKernel     = nullptr;
float *recurrentKernel = nullptr;
float *lstmBias        = nullptr;
float *denseKernel     = nullptr;
float *denseBias       = nullptr;

float *hiddenState = nullptr;
float *cellState   = nullptr;
float *gateVector  = nullptr;

float *inputWindow = nullptr;

// ---------------------------------------------------------------------
// Benchmark Results
// ---------------------------------------------------------------------

struct BenchmarkResults
{
  uint32_t modelLoadTimeUs;
  uint32_t rawDataLoadTimeUs;

  uint32_t minForwardPassUs;
  uint32_t maxForwardPassUs;

  uint64_t totalForwardPassUs;

  float averageForwardPassUs;
  float averagePerSampleUs;
  float windowsPerSecond;

  float lastOutput;

  float inputWindowFirstValue;
  float inputWindowLastValue;
  float inputWindowMean;
  float inputWindowMin;
  float inputWindowMax;

  size_t parameterCount;
  size_t modelParameterBytes;
  size_t runtimeWorkspaceBytes;
  size_t inputBufferBytes;
  size_t totalBenchmarkBytes;
};

BenchmarkResults results;

// ---------------------------------------------------------------------
// Stage Pulse
// ---------------------------------------------------------------------

void stagePulse(uint8_t stageNumber)
{
  Serial.print("Stage ");
  Serial.print(stageNumber);
  Serial.println(" marker pulse.");

  digitalWrite(TRIGGER_PIN, HIGH);
  delay(STAGE_PULSE_MS);
  digitalWrite(TRIGGER_PIN, LOW);
}

// ---------------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------------

static inline float sigmoidf_fast(float x)
{
  return 1.0f / (1.0f + expf(-x));
}

bool readNextFloat(File &file, float &value)
{
  char buffer[40];
  size_t index = 0;

  while (file.available())
  {
    char c = file.peek();

    if (c == ',' || c == '\n' || c == '\r' ||
        c == ' ' || c == '\t')
    {
      file.read();
    }
    else
    {
      break;
    }
  }

  if (!file.available())
    return false;

  while (file.available() && index < sizeof(buffer) - 1)
  {
    char c = file.peek();

    if (c == ',' || c == '\n' || c == '\r' ||
        c == ' ' || c == '\t')
    {
      break;
    }

    buffer[index++] = file.read();
  }

  buffer[index] = '\0';

  if (index == 0)
    return false;

  value = strtof(buffer, nullptr);
  return true;
}

bool loadCSV(const char *filename, float *destination, size_t count)
{
  Serial.print("  Loading ");
  Serial.print(filename);
  Serial.print(" ... ");

  File file = SD.open(filename, FILE_READ);

  if (!file)
  {
    Serial.println("FAILED TO OPEN");
    return false;
  }

  size_t loaded = 0;

  while (loaded < count)
  {
    float value;

    if (!readNextFloat(file, value))
      break;

    destination[loaded++] = value;
  }

  bool hasExtraValues = false;
  float extraValue;

  if (loaded == count)
  {
    hasExtraValues = readNextFloat(file, extraValue);
  }

  file.close();

  if (loaded != count)
  {
    Serial.print("FAILED - loaded ");
    Serial.print(loaded);
    Serial.print(" values, expected ");
    Serial.println(count);
    return false;
  }

  if (hasExtraValues)
  {
    Serial.print("FAILED - expected ");
    Serial.print(count);
    Serial.println(" values but file contains extra numeric values");
    return false;
  }

  Serial.print("OK (");
  Serial.print(loaded);
  Serial.println(" values)");

  return true;
}

// ---------------------------------------------------------------------
// Load RAW.CSV
// ---------------------------------------------------------------------

bool loadRawInputWindow()
{
  File file = SD.open(RAW_FILE, FILE_READ);

  if (!file)
    return false;

  size_t rowIndex = 0;
  size_t storedSamples = 0;

  while (file.available() && storedSamples < WINDOW_SIZE)
  {
    float timeValue;
    float accelerationValue;

    if (!readNextFloat(file, timeValue))
    {
      file.close();
      return false;
    }

    if (!readNextFloat(file, accelerationValue))
    {
      file.close();
      return false;
    }

    if (rowIndex >= RAW_START_SAMPLE)
    {
      inputWindow[storedSamples] = accelerationValue;
      storedSamples++;
    }

    rowIndex++;
  }

  file.close();

  return storedSamples == WINDOW_SIZE;
}

void calculateInputWindowStatistics()
{
  results.inputWindowFirstValue = inputWindow[0];
  results.inputWindowLastValue  = inputWindow[WINDOW_SIZE - 1];

  float sumValue = 0.0f;
  float minValue = inputWindow[0];
  float maxValue = inputWindow[0];

  for (int i = 0; i < WINDOW_SIZE; ++i)
  {
    const float x = inputWindow[i];

    sumValue += x;

    if (x < minValue)
      minValue = x;

    if (x > maxValue)
      maxValue = x;
  }

  results.inputWindowMean =
      sumValue / static_cast<float>(WINDOW_SIZE);

  results.inputWindowMin = minValue;
  results.inputWindowMax = maxValue;
}

// ---------------------------------------------------------------------
// Allocate Memory
// ---------------------------------------------------------------------

bool allocateMemory()
{
  const size_t inputKernelCount =
      INPUT_SIZE * GATE_SIZE;

  const size_t recurrentKernelCount =
      HIDDEN_SIZE * GATE_SIZE;

  const size_t biasCount =
      GATE_SIZE;

  const size_t denseKernelCount =
      HIDDEN_SIZE;

  inputKernel =
      (float *)malloc(inputKernelCount * sizeof(float));

  recurrentKernel =
      (float *)malloc(recurrentKernelCount * sizeof(float));

  lstmBias =
      (float *)malloc(biasCount * sizeof(float));

  denseKernel =
      (float *)malloc(denseKernelCount * sizeof(float));

  denseBias =
      (float *)malloc(sizeof(float));

  hiddenState =
      (float *)malloc(HIDDEN_SIZE * sizeof(float));

  cellState =
      (float *)malloc(HIDDEN_SIZE * sizeof(float));

  gateVector =
      (float *)malloc(GATE_SIZE * sizeof(float));

  inputWindow =
      (float *)malloc(WINDOW_SIZE * sizeof(float));

  if (!inputKernel ||
      !recurrentKernel ||
      !lstmBias ||
      !denseKernel ||
      !denseBias ||
      !hiddenState ||
      !cellState ||
      !gateVector ||
      !inputWindow)
  {
    return false;
  }

  return true;
}

// ---------------------------------------------------------------------
// Load Model
// ---------------------------------------------------------------------

bool loadModelFromSD()
{
  Serial.println();
  Serial.println("Checking individual model parameter files:");

  if (!loadCSV(FILE_W0, inputKernel, INPUT_SIZE * GATE_SIZE))
  {
    Serial.println("ERROR FILE: W0.CSV");
    return false;
  }

  if (!loadCSV(FILE_W1, recurrentKernel, HIDDEN_SIZE * GATE_SIZE))
  {
    Serial.println("ERROR FILE: W1.CSV");
    return false;
  }

  if (!loadCSV(FILE_B, lstmBias, GATE_SIZE))
  {
    Serial.println("ERROR FILE: B.CSV");
    return false;
  }

  if (!loadCSV(FILE_FC, denseKernel, HIDDEN_SIZE))
  {
    Serial.println("ERROR FILE: FCW.CSV");
    return false;
  }

  if (!loadCSV(FILE_FB, denseBias, 1))
  {
    Serial.println("ERROR FILE: FCB.CSV");
    return false;
  }

  Serial.println("All five model parameter files passed validation.");
  return true;
}

// ---------------------------------------------------------------------
// LSTM Forward Pass
// ---------------------------------------------------------------------

float runForwardPass(const float *window)
{
  for (int h = 0; h < HIDDEN_SIZE; ++h)
  {
    hiddenState[h] = 0.0f;
    cellState[h]   = 0.0f;
  }

  for (int t = 0; t < WINDOW_SIZE; ++t)
  {
    const float x = window[t];

    for (int g = 0; g < GATE_SIZE; ++g)
    {
      gateVector[g] =
          lstmBias[g] +
          inputKernel[g] * x;
    }

    for (int h = 0; h < HIDDEN_SIZE; ++h)
    {
      const float previousHidden =
          hiddenState[h];

      const int rowOffset =
          h * GATE_SIZE;

      for (int g = 0; g < GATE_SIZE; ++g)
      {
        gateVector[g] +=
            previousHidden *
            recurrentKernel[rowOffset + g];
      }
    }

    for (int h = 0; h < HIDDEN_SIZE; ++h)
    {
      const float inputGate =
          sigmoidf_fast(
              gateVector[h]);

      const float forgetGate =
          sigmoidf_fast(
              gateVector[HIDDEN_SIZE + h]);

      const float candidate =
          tanhf(
              gateVector[2 * HIDDEN_SIZE + h]);

      const float outputGate =
          sigmoidf_fast(
              gateVector[3 * HIDDEN_SIZE + h]);

      cellState[h] =
          forgetGate * cellState[h] +
          inputGate * candidate;

      hiddenState[h] =
          outputGate * tanhf(cellState[h]);
    }
  }

  float output = denseBias[0];

  for (int h = 0; h < HIDDEN_SIZE; ++h)
  {
    output += hiddenState[h] * denseKernel[h];
  }

  return output;
}

// ---------------------------------------------------------------------
// Memory Footprint
// ---------------------------------------------------------------------

void calculateMemoryFootprint()
{
  const size_t inputKernelBytes =
      INPUT_SIZE * GATE_SIZE * sizeof(float);

  const size_t recurrentKernelBytes =
      HIDDEN_SIZE * GATE_SIZE * sizeof(float);

  const size_t biasBytes =
      GATE_SIZE * sizeof(float);

  const size_t denseKernelBytes =
      HIDDEN_SIZE * sizeof(float);

  const size_t denseBiasBytes =
      sizeof(float);

  results.modelParameterBytes =
      inputKernelBytes +
      recurrentKernelBytes +
      biasBytes +
      denseKernelBytes +
      denseBiasBytes;

  results.parameterCount =
      results.modelParameterBytes / sizeof(float);

  const size_t hiddenBytes =
      HIDDEN_SIZE * sizeof(float);

  const size_t cellBytes =
      HIDDEN_SIZE * sizeof(float);

  const size_t gateBytes =
      GATE_SIZE * sizeof(float);

  results.runtimeWorkspaceBytes =
      hiddenBytes +
      cellBytes +
      gateBytes;

  results.inputBufferBytes =
      WINDOW_SIZE * sizeof(float);

  results.totalBenchmarkBytes =
      results.modelParameterBytes +
      results.runtimeWorkspaceBytes +
      results.inputBufferBytes;
}

// ---------------------------------------------------------------------
// Warm-Up Passes
// ---------------------------------------------------------------------

void warmupForwardPasses()
{
  volatile float temporaryOutput = 0.0f;

  for (int i = 0; i < WARMUP_RUNS; ++i)
  {
    temporaryOutput = runForwardPass(inputWindow);
  }
}

// ---------------------------------------------------------------------
// Timed Forward-Pass Benchmark
//
// NO Serial and NO SD I/O inside this function.
// ---------------------------------------------------------------------

void benchmarkForwardPass()
{
  results.totalForwardPassUs = 0;
  results.minForwardPassUs = UINT32_MAX;
  results.maxForwardPassUs = 0;

  volatile float lastOutput = 0.0f;

  for (int run = 0; run < BENCHMARK_RUNS; ++run)
  {
    const uint32_t startUs = micros();

    lastOutput = runForwardPass(inputWindow);

    const uint32_t elapsedUs = micros() - startUs;

    results.totalForwardPassUs += elapsedUs;

    if (elapsedUs < results.minForwardPassUs)
      results.minForwardPassUs = elapsedUs;

    if (elapsedUs > results.maxForwardPassUs)
      results.maxForwardPassUs = elapsedUs;
  }

  results.averageForwardPassUs =
      static_cast<float>(results.totalForwardPassUs) /
      static_cast<float>(BENCHMARK_RUNS);

  results.averagePerSampleUs =
      results.averageForwardPassUs /
      static_cast<float>(WINDOW_SIZE);

  results.windowsPerSecond =
      1.0e6f /
      results.averageForwardPassUs;

  results.lastOutput = lastOutput;
}

// ---------------------------------------------------------------------
// Save Results
// ---------------------------------------------------------------------

bool saveResultsToSD()
{
  if (SD.exists(RESULT_FILE))
  {
    SD.remove(RESULT_FILE);
  }

  File file = SD.open(RESULT_FILE, FILE_WRITE);

  if (!file)
    return false;

  file.println(
      "ModelUnits,"
      "WindowSize,"
      "RawStartSample,"
      "WarmupRuns,"
      "BenchmarkRuns,"
      "ParameterCount,"
      "ModelParameterBytes,"
      "RuntimeWorkspaceBytes,"
      "InputBufferBytes,"
      "TotalBenchmarkBytes,"
      "ModelLoadTime_us,"
      "RawDataLoadTime_us,"
      "AverageForwardPass_us,"
      "MinimumForwardPass_us,"
      "MaximumForwardPass_us,"
      "AveragePerSample_us,"
      "WindowsPerSecond,"
      "InputFirstValue,"
      "InputLastValue,"
      "InputMean,"
      "InputMin,"
      "InputMax,"
      "LastOutput");

  file.print(HIDDEN_SIZE);
  file.print(',');

  file.print(WINDOW_SIZE);
  file.print(',');

  file.print(RAW_START_SAMPLE);
  file.print(',');

  file.print(WARMUP_RUNS);
  file.print(',');

  file.print(BENCHMARK_RUNS);
  file.print(',');

  file.print(results.parameterCount);
  file.print(',');

  file.print(results.modelParameterBytes);
  file.print(',');

  file.print(results.runtimeWorkspaceBytes);
  file.print(',');

  file.print(results.inputBufferBytes);
  file.print(',');

  file.print(results.totalBenchmarkBytes);
  file.print(',');

  file.print(results.modelLoadTimeUs);
  file.print(',');

  file.print(results.rawDataLoadTimeUs);
  file.print(',');

  file.print(results.averageForwardPassUs, 6);
  file.print(',');

  file.print(results.minForwardPassUs);
  file.print(',');

  file.print(results.maxForwardPassUs);
  file.print(',');

  file.print(results.averagePerSampleUs, 9);
  file.print(',');

  file.print(results.windowsPerSecond, 6);
  file.print(',');

  file.print(results.inputWindowFirstValue, 9);
  file.print(',');

  file.print(results.inputWindowLastValue, 9);
  file.print(',');

  file.print(results.inputWindowMean, 9);
  file.print(',');

  file.print(results.inputWindowMin, 9);
  file.print(',');

  file.print(results.inputWindowMax, 9);
  file.print(',');

  file.println(results.lastOutput, 9);

  file.flush();
  file.close();

  return true;
}

// ---------------------------------------------------------------------
// Halt Functions
// ---------------------------------------------------------------------

void haltWithError(const char *message)
{
  digitalWrite(TRIGGER_PIN, LOW);
  digitalWrite(LED_PIN, LOW);

  Serial.print("ERROR: ");
  Serial.println(message);
  Serial.println("Benchmark halted.");

  while (true)
  {
    delay(1000);
  }
}

void haltSuccess()
{
  digitalWrite(TRIGGER_PIN, LOW);
  digitalWrite(LED_PIN, LOW);

  while (true)
  {
    delay(1000);
  }
}

// ---------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------

void setup()
{
  // Package LED ON for the full iteration.
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);

  // Trigger starts LOW. No startup HIGH state.
  pinMode(TRIGGER_PIN, OUTPUT);
  digitalWrite(TRIGGER_PIN, LOW);

  // Serial diagnostics.
  // This wait occurs while TRIGGER_PIN remains LOW.
  Serial.begin(115200);

  const uint32_t serialWaitStart = millis();

  while (!Serial && (millis() - serialWaitStart < 3000))
  {
  }

  Serial.println();
  Serial.println("==============================================");
  Serial.println("LSTM 6-Unit Benchmark Starting");
  Serial.println("Exactly six 300 ms trigger pulses expected.");
  Serial.println("==============================================");

  // ---------------------------------------------------------------
  // Stage 1: SD initialization
  // ---------------------------------------------------------------

  stagePulse(1);

  Serial.println("Initializing SD card...");

  if (!SD.begin(SD_CS_PIN))
  {
    haltWithError("SD card initialization failed.");
  }

  Serial.println("SD card initialized.");

  // ---------------------------------------------------------------
  // Memory allocation
  //
  // This is intentionally not assigned its own stage pulse because
  // the requested six stages do not include memory allocation.
  // ---------------------------------------------------------------

  Serial.println("Allocating memory...");

  if (!allocateMemory())
  {
    haltWithError("Model/runtime memory allocation failed.");
  }

  Serial.println("Memory allocation successful.");

  // ---------------------------------------------------------------
  // Stage 2: Model-weight loading
  // ---------------------------------------------------------------

  stagePulse(2);

  Serial.println("Loading model parameter files...");

  const uint32_t modelLoadStartUs = micros();

  if (!loadModelFromSD())
  {
    haltWithError(
        "Model parameter loading failed. See per-file diagnostic above.");
  }

  results.modelLoadTimeUs =
      micros() - modelLoadStartUs;

  Serial.println("Model parameters loaded.");

  // ---------------------------------------------------------------
  // Stage 3: RAW.CSV loading
  // ---------------------------------------------------------------

  stagePulse(3);

  Serial.println("Loading RAW.CSV input window...");

  const uint32_t rawLoadStartUs = micros();

  if (!loadRawInputWindow())
  {
    haltWithError(
        "RAW.CSV loading failed. Check filename and sample count.");
  }

  results.rawDataLoadTimeUs =
      micros() - rawLoadStartUs;

  calculateInputWindowStatistics();

  Serial.println("RAW.CSV loaded.");

  // Calculate memory footprint before forward-pass tests.
  calculateMemoryFootprint();

  // ---------------------------------------------------------------
  // Stage 4: Warm-up forward passes
  // ---------------------------------------------------------------

  stagePulse(4);

  Serial.println("Starting warm-up passes...");

  warmupForwardPasses();

  Serial.println("Warm-up passes complete.");

  // ---------------------------------------------------------------
  // Stage 5: Timed repeated LSTM forward passes
  // ---------------------------------------------------------------

  stagePulse(5);

  Serial.println(
      "Starting timed benchmark "
      "(no Serial/SD I/O inside timed loop)...");

  benchmarkForwardPass();

  Serial.println("Benchmark complete.");

  Serial.print("Average forward pass (us/window): ");
  Serial.println(results.averageForwardPassUs, 6);

  Serial.print("Minimum forward pass (us/window): ");
  Serial.println(results.minForwardPassUs);

  Serial.print("Maximum forward pass (us/window): ");
  Serial.println(results.maxForwardPassUs);

  // ---------------------------------------------------------------
  // Stage 6: Write BM6.CSV
  // ---------------------------------------------------------------

  stagePulse(6);

  Serial.println("Writing BM6.CSV...");

  if (!saveResultsToSD())
  {
    haltWithError("Could not create/write BM6.CSV.");
  }

  Serial.println("BM6.CSV saved successfully.");
  Serial.println("Six stage pulses completed.");
  Serial.println("Turning LED OFF and halting.");

  // LED goes OFF only after result writing is complete.
  haltSuccess();
}

// ---------------------------------------------------------------------
// Loop
// ---------------------------------------------------------------------

void loop()
{
  // Never reached.
}

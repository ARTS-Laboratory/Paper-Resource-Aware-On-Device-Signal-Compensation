/*
  ======================================================================
  INA219 Six-Pulse Current Logger
  ======================================================================

  Purpose
  -------
  Records current draw continuously across a six-pulse digital trigger
  sequence.

  Logging behavior:
    1) Wait for the FIRST rising edge of the trigger.
    2) Create a new CSV file.
    3) Start time at 0.
    4) Continuously log:
           Time_s,Trigger,Current_mA
    5) Keep logging during BOTH HIGH pulses and LOW gaps.
    6) Count each completed pulse on the falling edge.
    7) After the SIXTH falling edge, write the final LOW-state sample,
       close the file, and wait for the next six-pulse sequence.

  Output files:
      I000.CSV
      I001.CSV
      I002.CSV
      ...

  Required libraries:
      Wire
      Adafruit INA219
      SD
      SPI
  ======================================================================
*/

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_INA219.h>
#include <SD.h>
#include <SPI.h>

// ---------------------------------------------------------------------
// User Settings
// ---------------------------------------------------------------------

constexpr uint8_t TRIGGER_PIN = 2;
constexpr uint8_t SD_CS_PIN   = 4;
constexpr uint8_t LED_PIN     = 3;

// Number of complete trigger pulses in one test.
constexpr uint8_t TOTAL_PULSES = 6;

// Logging interval.
// 10,000 us = 200 samples/second.
constexpr uint32_t SAMPLE_INTERVAL_US = 20000;

// ---------------------------------------------------------------------
// Hardware
// ---------------------------------------------------------------------

Adafruit_INA219 ina219;
File dataFile;

// ---------------------------------------------------------------------
// Logger State
// ---------------------------------------------------------------------

bool recording = false;

bool previousTriggerState = LOW;
bool currentTriggerState  = LOW;

uint8_t completedPulses = 0;

uint32_t measurementStartUs = 0;
uint32_t previousSampleUs   = 0;

char fileName[13];

// ---------------------------------------------------------------------
// Find Next Available CSV Filename
// ---------------------------------------------------------------------

bool getNextFileName()
{
  for (int i = 0; i < 1000; ++i)
  {
    snprintf(fileName, sizeof(fileName), "I%03d.CSV", i);

    if (!SD.exists(fileName))
    {
      return true;
    }
  }

  return false;
}

// ---------------------------------------------------------------------
// Write One Measurement
// ---------------------------------------------------------------------

void writeMeasurement(uint32_t sampleTimeUs, bool triggerState)
{
  if (!recording || !dataFile)
    return;

  const float elapsedTime =
      static_cast<float>(sampleTimeUs - measurementStartUs) / 1000000.0f;

  const float current_mA = ina219.getCurrent_mA();

  dataFile.print(elapsedTime, 6);
  dataFile.print(',');
  dataFile.print(triggerState ? 1 : 0);
  dataFile.print(',');
  dataFile.println(current_mA, 4);
}

// ---------------------------------------------------------------------
// Start Logging
// ---------------------------------------------------------------------

bool startRecording()
{
  if (!getNextFileName())
  {
    Serial.println("ERROR: No available filenames I000.CSV-I999.CSV.");
    return false;
  }

  dataFile = SD.open(fileName, FILE_WRITE);

  if (!dataFile)
  {
    Serial.print("ERROR: Could not create ");
    Serial.println(fileName);
    return false;
  }

  dataFile.println("Time_s,Trigger,Current_mA");

  completedPulses = 0;

  measurementStartUs = micros();
  previousSampleUs   = measurementStartUs;

  recording = true;

  digitalWrite(LED_PIN, HIGH);

  Serial.print("Logging started: ");
  Serial.println(fileName);

  // Record the first HIGH state immediately at approximately t = 0.
  writeMeasurement(measurementStartUs, HIGH);

  return true;
}

// ---------------------------------------------------------------------
// Stop Logging
// ---------------------------------------------------------------------

void stopRecording()
{
  if (dataFile)
  {
    dataFile.flush();
    dataFile.close();
  }

  recording = false;
  digitalWrite(LED_PIN, LOW);

  Serial.print("Six pulses completed. Saved: ");
  Serial.println(fileName);
  Serial.println("Waiting for next pulse sequence...");
}

// ---------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------

void setup()
{
  Serial.begin(9600);

  pinMode(TRIGGER_PIN, INPUT);
  pinMode(LED_PIN, OUTPUT);

  digitalWrite(LED_PIN, LOW);

  // ---------------------------------------------------------------
  // INA219
  // ---------------------------------------------------------------

  Wire.begin();

  Serial.println("Initializing INA219...");

  if (!ina219.begin())
  {
    Serial.println("ERROR: INA219 not found.");

    while (true)
    {
      digitalWrite(LED_PIN, HIGH);
      delay(100);
      digitalWrite(LED_PIN, LOW);
      delay(100);
    }
  }

  Serial.println("INA219 initialized.");

  // ---------------------------------------------------------------
  // SD Card
  // ---------------------------------------------------------------

  Serial.println("Initializing SD card...");

  if (!SD.begin(SD_CS_PIN))
  {
    Serial.println("ERROR: SD card initialization failed.");

    while (true)
    {
      digitalWrite(LED_PIN, HIGH);
      delay(500);
      digitalWrite(LED_PIN, LOW);
      delay(500);
    }
  }

  Serial.println("SD card initialized.");

  // Read initial trigger condition before entering loop.
  previousTriggerState = digitalRead(TRIGGER_PIN);

  Serial.println("Waiting for first trigger pulse...");
}

// ---------------------------------------------------------------------
// Main Loop
// ---------------------------------------------------------------------

void loop()
{
  currentTriggerState = digitalRead(TRIGGER_PIN);

  // ===============================================================
  // Detect FIRST rising edge and start the six-pulse recording
  // ===============================================================

  if (!recording &&
      previousTriggerState == LOW &&
      currentTriggerState == HIGH)
  {
    if (!startRecording())
    {
      delay(100);
    }
  }

  // ===============================================================
  // Continuous Current Logging
  //
  // This continues during both:
  //      Trigger HIGH
  //      Trigger LOW
  //
  // so the complete six-pulse sequence is captured.
  // ===============================================================

  if (recording)
  {
    const uint32_t nowUs = micros();

    if ((uint32_t)(nowUs - previousSampleUs) >= SAMPLE_INTERVAL_US)
    {
      previousSampleUs += SAMPLE_INTERVAL_US;

      writeMeasurement(nowUs, currentTriggerState);
    }

    // =============================================================
    // Count completed pulses using falling edges
    // =============================================================

    if (previousTriggerState == HIGH &&
        currentTriggerState == LOW)
    {
      completedPulses++;

      Serial.print("Completed pulse ");
      Serial.print(completedPulses);
      Serial.print(" of ");
      Serial.println(TOTAL_PULSES);

      // Write the falling edge immediately so the CSV explicitly
      // contains the transition to Trigger = 0.
      writeMeasurement(micros(), LOW);

      // ===========================================================
      // Stop AFTER the sixth pulse has gone LOW
      // ===========================================================

      if (completedPulses >= TOTAL_PULSES)
      {
        stopRecording();
      }
    }
  }

  previousTriggerState = currentTriggerState;
}

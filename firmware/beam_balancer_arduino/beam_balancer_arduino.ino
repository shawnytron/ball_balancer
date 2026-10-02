#include <Servo.h>

// J1 on the supplied PCB: 1=5V, 2=DA, 3=DB, 4=DC, 5=ADC_IN, 6=GND.
// Elegoo Mega 2560: use the even-numbered side of the 22-53 digital header.
// These are board labels, not ATmega chip pin numbers. Verify J1 orientation.
const byte SELECT_A = 22;
const byte SELECT_B = 24;
const byte SELECT_C = 26;
const byte SENSOR_ADC = A0;
const byte SERVO_SIGNAL = 28;  // Servo library works on this ordinary digital pin.

// Protocol v3: A commands accept decimal degrees (up to three decimal places).
// Python must configure the operating range before any movement.
// Limits live in beam_config.json and are sent again after each board reset.
int servoNeutral = 45;
int servoMin = 26;
int servoMax = 70;
long commandedMilliDegrees = 45000;
bool limitsConfigured = false;
const unsigned long COMMAND_TIMEOUT_MS = 400;
const unsigned long SENSOR_PERIOD_MS = 20;

// Match the default mapping used for the original integer-degree calibration.
// Do not substitute a 1000..2000 us mapping: that would change taught positions.
const int SERVO_ZERO_US = 544;
const int SERVO_180_US = 2400;
Servo beamServo;  // Global zero-initialization also sets its pre-attach bounds.
bool servoAttached = false;
unsigned long lastCommandMs = 0;
unsigned long lastSensorMs = 0;
char command[40];
byte commandLength = 0;
bool commandOverflow = false;

int degreesToPulseUs(long milliDegrees) {
  // Preserve every old Servo.write(integer) pulse exactly. Interpolate only
  // between adjacent integer-degree pulses, then round to the nearest us.
  int whole = milliDegrees / 1000;
  int fraction = milliDegrees % 1000;
  long pulseLow = SERVO_ZERO_US + (long)(SERVO_180_US - SERVO_ZERO_US) * whole / 180;
  if (!fraction) return (int)pulseLow;
  long pulseHigh = SERVO_ZERO_US + (long)(SERVO_180_US - SERVO_ZERO_US) * (whole + 1) / 180;
  return (int)(pulseLow + ((pulseHigh - pulseLow) * fraction + 500) / 1000);
}

void setServo(long milliDegrees) {
  if (!limitsConfigured) return;
  long safeAngle = constrain(milliDegrees, (long)servoMin * 1000, (long)servoMax * 1000);
  int pulse = degreesToPulseUs(safeAngle);
  // Pulse limits are enforced independently as well as angle limits.
  pulse = constrain(pulse, degreesToPulseUs((long)servoMin * 1000),
                    degreesToPulseUs((long)servoMax * 1000));
  beamServo.writeMicroseconds(pulse);  // Store target before enabling first pulse.
  if (!servoAttached) {
    beamServo.attach(SERVO_SIGNAL, SERVO_ZERO_US, SERVO_180_US);
    servoAttached = true;
  }
  commandedMilliDegrees = safeAngle;
  lastCommandMs = millis();
}

void returnToNeutral() {
  if (servoAttached) setServo((long)servoNeutral * 1000);
}

bool parseAngle(const char *text, long &milliDegrees) {
  // Fixed-point parser: no NaN/Inf, signs, exponents, trailing garbage, or
  // overflow; accepting only the decimal format the Python client transmits.
  if (*text < '0' || *text > '9') return false;
  long whole = 0;
  while (*text >= '0' && *text <= '9') {
    whole = whole * 10 + (*text++ - '0');
    if (whole > 180) return false;
  }
  int fraction = 0;
  int scale = 100;
  if (*text == '.') {
    ++text;
    if (*text < '0' || *text > '9') return false;
    while (*text >= '0' && *text <= '9') {
      if (!scale) return false;
      fraction += (*text++ - '0') * scale;
      scale /= 10;
    }
  }
  if (*text != '\0' || (whole == 180 && fraction)) return false;
  milliDegrees = whole * 1000 + fraction;
  return true;
}

bool parseLimits(const char *text, int &low, int &level, int &high) {
  long values[3];
  for (byte i = 0; i < 3; ++i) {
    char *end;
    values[i] = strtol(text, &end, 10);
    if (end == text || values[i] < 0 || values[i] > 180) return false;
    if (i < 2) {
      if (*end != ',') return false;
      text = end + 1;
    } else if (*end != '\0') return false;
  }
  low = values[0]; level = values[1]; high = values[2];
  return low < level && level < high;
}

void handleCommand(const char *line) {
  if (strcmp(line, "HELLO") == 0) {
    Serial.println("V,3");
    return;
  }
  if (line[0] == 'L' && line[1] == ',') {
    int low, level, high;
    if (!parseLimits(line + 2, low, level, high) ||
        (servoAttached && (commandedMilliDegrees < (long)low * 1000 || commandedMilliDegrees > (long)high * 1000))) {
      Serial.println("ERR,LIMITS");
      return;
    }
    servoMin = low; servoNeutral = level; servoMax = high;
    limitsConfigured = true;
    Serial.print("L,"); Serial.print(low); Serial.print(',');
    Serial.print(level); Serial.print(','); Serial.println(high);
    return;
  }
  if (strcmp(line, "STOP") == 0) {
    returnToNeutral();
    return;
  }
  if (line[0] != 'A' || line[1] != ',') return;
  long milliDegrees;
  if (!parseAngle(line + 2, milliDegrees)) return;
  setServo(milliDegrees);
}

void readSerialCommands() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (commandLength && !commandOverflow) {
        command[commandLength] = '\0';
        handleCommand(command);
      }
      commandLength = 0;
      commandOverflow = false;
    } else if (!commandOverflow && commandLength < sizeof(command) - 1) {
      command[commandLength++] = c;
    } else {
      commandOverflow = true;  // Discard the entire line, including its suffix.
    }
  }
}

int readSensor(byte channel) {
  digitalWrite(SELECT_A, channel & 1);
  digitalWrite(SELECT_B, (channel >> 1) & 1);
  digitalWrite(SELECT_C, (channel >> 2) & 1);
  delayMicroseconds(100);
  analogRead(SENSOR_ADC);  // Discard the first sample after switching channels.
  return analogRead(SENSOR_ADC);
}

void setup() {
  pinMode(SELECT_A, OUTPUT);
  pinMode(SELECT_B, OUTPUT);
  pinMode(SELECT_C, OUTPUT);
  Serial.begin(115200);
  // Servo stays detached until a valid command is received.
}

void loop() {
  readSerialCommands();
  unsigned long now = millis();
  if (servoAttached && now - lastCommandMs > COMMAND_TIMEOUT_MS) {
    returnToNeutral();
  }
  if (now - lastSensorMs >= SENSOR_PERIOD_MS) {
    lastSensorMs = now;
    Serial.print("S,");
    Serial.print(now);
    for (byte i = 0; i < 8; ++i) {
      Serial.print(',');
      Serial.print(readSensor(i));
    }
    Serial.println();
  }
}

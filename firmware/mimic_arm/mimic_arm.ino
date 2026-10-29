// mimic_arm.ino: firmware for the KEYESTUDIO ESP32 arm (4 servos: base, shoulder, elbow, grip).
//
// One line in, one line out over USB serial at 115200 baud (docs/plan.md 2.2):
//   J <base> <shoulder> <elbow> <grip> <ms>   smooth move        -> OK, or ERR <reason>
//   H                                         go home            -> OK
//   ?                                         where are you?     -> A <base> <shoulder> <elbow> <grip>
//   S                                         stop and hold      -> OK
//
// Every safety rule (clamps, speed limit, grip timer, placeholder lock) is in arm_logic.h, which is
// tested on the laptop. This file only reads the serial port and drives the servos.
//
// Status 29 Oct: written but NOT compiled for the ESP32 and NOT uploaded. Pins and limits are
// placeholders until the arm is measured (needs.md B4).
//
// Power-on: a hobby servo cannot report its position, so attach() makes every joint jump to HOME.
// Keep HOME a safe pose, and always send H before switching off so the jump at the next start is small.
#include <ESP32Servo.h>
#include "arm_logic.h"

Servo servos[4];
ArmState arm;
char line[MAX_LINE + 1];
int line_len = 0;
bool line_too_long = false;
unsigned long last_tick = 0;

void write_servos() {
  for (int j = 0; j < 4; j++) servos[j].write((int)lroundf(arm.now[j]));
}

void setup() {
  Serial.begin(115200);
  arm_init(arm);
  for (int j = 0; j < 4; j++) {
    servos[j].setPeriodHertz(50);
    servos[j].attach(PINS[j], 500, 2400);
  }
  write_servos();
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      line[line_len] = '\0';
      char reply[48];
      if (line_too_long) {
        snprintf(reply, sizeof reply, "ERR line too long");
      } else {
        handle_line(arm, line, millis(), reply, sizeof reply);
      }
      Serial.println(reply);
      line_len = 0;
      line_too_long = false;
    } else if (line_len < MAX_LINE) {
      line[line_len++] = c;
    } else {
      line_too_long = true;  // keep reading to the end of the line, then reject it
    }
  }

  unsigned long now = millis();
  if (now - last_tick >= 20) {
    last_tick = now;
    tick(arm, now);
    write_servos();  // writes where the arm is now; with no move running that is the same angle again
  }
}

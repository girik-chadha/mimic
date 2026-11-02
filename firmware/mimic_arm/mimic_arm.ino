// mimic_arm.ino: firmware for the KEYESTUDIO ESP32 arm (4 servos: base, shoulder, elbow, grip).
//
// One line in, one line out over USB serial at 115200 baud (docs/plan.md 2.2):
//   J <base> <shoulder> <elbow> <grip> <ms>   smooth move        -> OK, or ERR <reason>
//   H                                         go home            -> OK
//   ?                                         where are you?     -> A <base> <shoulder> <elbow> <grip>
//                                                                   (with " L" on the end while locked)
//   S                                         stop and hold      -> OK
//
// Every safety rule (clamps, speed limit, grip timer, placeholder lock) is in arm_logic.h, which is
// tested on the laptop. This file only reads the serial port and drives the servos.
//
// Status 3 Nov: UNLOCKED with the kit's ranges (from its sample code, untested on this arm) for the first powered
// session (needs.md C7). Compiles for esp32:esp32:esp32 (ESP32 core 2.0.12, ESP32Servo 1.2.1); not uploaded yet,
// so the board still runs the locked build. Pins are from the kit's sample code (untested on this arm).
// The servo ports need battery power: USB alone does not move them.
//
// While locked (placeholder limits, or not checked by hand) no servo is attached and no angle is ever
// written: the servos get no signal. Only ? and S work; J and H reply ERR limits not measured.
// Once unlocked, at start-up the servos are switched on one at a time, 400 ms apart (base, shoulder, elbow,
// claw: servos_to_attach() in arm_logic.h). A hobby servo cannot report its position, so each one jumps to HOME
// as it is switched on. Keep HOME a safe pose, and always send H before switching off so the next jump is small.
// Every J and H after that is ramped (arm_logic.h: start_move and tick).
#include <ESP32Servo.h>
#include "arm_logic.h"

Servo servos[4];
ArmState arm;
char line[MAX_LINE + 1];
int line_len = 0;
bool line_too_long = false;
unsigned long last_tick = 0;
unsigned long started_ms = 0;
int attached = 0;  // how many servos are switched on so far (base first, claw last)

// Switch on any servo that is due (servos_to_attach is 0 while locked, so then this never attaches anything).
void attach_due_servos() {
  int due = servos_to_attach(millis() - started_ms);
  while (attached < due) {
    servos[attached].setPeriodHertz(50);
    servos[attached].attach(PINS[attached], 500, 2400);
    servos[attached].write((int)lroundf(arm.now[attached]));
    attached++;
  }
}

void write_servos() {
  if (!servos_enabled()) return;  // locked: never write an angle
  for (int j = 0; j < attached; j++) servos[j].write((int)lroundf(arm.now[j]));
}

void setup() {
  Serial.begin(115200);
  arm_init(arm);
  started_ms = millis();
  attach_due_servos();  // the base now; the others follow from loop(), 400 ms apart
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
    attach_due_servos();
    tick(arm, now);
    write_servos();  // writes where the arm is now; with no move running that is the same angle again
  }
}

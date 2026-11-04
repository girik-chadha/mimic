// mimic_arm.ino: firmware for the KEYESTUDIO ESP32 arm (4 servos: base, shoulder, elbow, grip).
//
// One line in, one line out over USB serial at 115200 baud (docs/plan.md 2.2):
//   J <base> <shoulder> <elbow> <grip> <ms>   smooth move        -> OK, or ERR <reason>
//   H                                         go home            -> OK
//   ?                                         where are you?     -> A <base> <shoulder> <elbow> <grip>
//                                                                   (" L" on the end while locked,
//                                                                    " D" while disarmed)
//   S                                         stop and hold      -> OK
//   E [<base> <shoulder> <elbow> <grip>]      enable (arm)       -> OK: switch the servos on, starting at these
//                                                                   angles (default 90 each)
//   D                                         disable            -> OK: stop, then switch every servo off (limp)
//   B                                         why did you start? -> B <reason> <seconds since start-up>
//                                                                   (power-on, brownout, software reset, watchdog, other)
//
// Every safety rule (clamps, speed limit, grip timer, placeholder lock) is in arm_logic.h, which is
// tested on the laptop. This file only reads the serial port and drives the servos.
//
// Status 3 Nov: UNLOCKED with the kit's ranges (from its sample code, untested on this arm) for the first powered
// session (needs.md C7), with B, and disarmed at power-on (E and D). Grip: a lower angle closes (4 Nov), open 90,
// closed limit 3 (PROVISIONAL), range 0-140. Compiles for
// esp32:esp32:esp32 (ESP32 core 2.0.12, ESP32Servo 1.2.1); not uploaded yet, so the board still runs the locked
// build (which has no B, E or D). Pins are from the kit's sample code (untested on this arm).
// The servo ports need battery power: USB alone does not move them.
//
// While locked (placeholder limits, or not checked by hand) no servo is attached and no angle is ever
// written: the servos get no signal. Only ?, S, B and D work; J, H and E reply ERR limits not measured.
// Once unlocked, the board still starts DISARMED: nothing is attached and no signal is sent, so switching on (or a
// restart) never moves the arm. E switches the servos on one at a time, 400 ms apart (base, shoulder, elbow, claw:
// servos_to_attach() in arm_logic.h), each starting at the angle E gives (default 90). A hobby servo cannot report
// its position, so each one jumps to that angle as it is switched on: pose the arm there by hand first. J and H
// reply ERR not enabled until all four are on. Every J and H is ramped (arm_logic.h: start_move and tick).
// D stops the arm and switches every servo off, so it goes limp (and sags under its own weight).
#include <ESP32Servo.h>
#include <esp_system.h>
#include "arm_logic.h"

// The reset codes in arm_logic.h must be the ESP32's own numbers: the build stops here if they ever differ.
static_assert(RESET_POWERON == ESP_RST_POWERON, "RESET_POWERON is not ESP_RST_POWERON");
static_assert(RESET_SW == ESP_RST_SW, "RESET_SW is not ESP_RST_SW");
static_assert(RESET_INT_WDT == ESP_RST_INT_WDT, "RESET_INT_WDT is not ESP_RST_INT_WDT");
static_assert(RESET_TASK_WDT == ESP_RST_TASK_WDT, "RESET_TASK_WDT is not ESP_RST_TASK_WDT");
static_assert(RESET_WDT == ESP_RST_WDT, "RESET_WDT is not ESP_RST_WDT");
static_assert(RESET_BROWNOUT == ESP_RST_BROWNOUT, "RESET_BROWNOUT is not ESP_RST_BROWNOUT");

Servo servos[4];
ArmState arm;
char line[MAX_LINE + 1];
int line_len = 0;
bool line_too_long = false;
unsigned long last_tick = 0;
int switched_on = 0;  // how many servos are attached right now (base first, claw last)

// Make the servos match arm.attached: switch on any that are due, switch off (detach, limp) any that are not.
// arm.attached is 0 while locked and while disarmed, so then nothing is ever attached.
void sync_servos() {
  while (switched_on < arm.attached) {
    servos[switched_on].setPeriodHertz(50);
    servos[switched_on].attach(PINS[switched_on], 500, 2400);
    servos[switched_on].write((int)lroundf(arm.now[switched_on]));
    switched_on++;
  }
  while (switched_on > arm.attached) {
    switched_on--;
    servos[switched_on].detach();  // no more signal: the servo goes limp
  }
}

void write_servos() {
  if (!servos_enabled()) return;  // locked: never write an angle
  for (int j = 0; j < switched_on; j++) servos[j].write((int)lroundf(arm.now[j]));
}

void setup() {
  Serial.begin(115200);
  arm_init(arm);  // disarmed: nothing is attached until E
  arm.boot_code = (int)esp_reset_reason();  // read once: why this start-up happened (the B command reports it)
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
    update_attached(arm, now);
    sync_servos();
    tick(arm, now);
    write_servos();  // writes where the arm is now; with no move running that is the same angle again
  }
}

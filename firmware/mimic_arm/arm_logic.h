// arm_logic.h: the arm's safety rules and move maths, in plain C++.
//
// No Arduino calls in this file, so exactly the same code runs on the ESP32 and is compiled and
// tested on the laptop (tests/unit/firmware_logic_test.cpp, run by tests/unit/test_firmware_logic.py).
// mimic_arm.ino only connects it to the servos and the serial port.
#pragma once
#include <math.h>
#include <stdio.h>
#include <string.h>

// ---------- limits: MUST match calib/arm_limits.json (tests/unit/test_arm.py checks) ----------
// KIT RANGES (3 Nov) from the kit's sample code, untested on this arm: for the first powered session only
// (needs.md C7), until scripts/first_moves.py has proposed measured ranges.
// Moves are refused unless LIMITS_ARE_PLACEHOLDER is 0 AND LIMITS_CHECKED_BY_HAND is 1.
#ifndef LIMITS_ARE_PLACEHOLDER
#define LIMITS_ARE_PLACEHOLDER 0
#endif
#ifndef LIMITS_CHECKED_BY_HAND
#define LIMITS_CHECKED_BY_HAND 1
#endif
#define GRIP_OPEN 90               // open, and where the 5 s relaxation returns the grip
// The closed limit is the ONE setting for the claw's direction: below GRIP_OPEN means a lower angle closes the claw
// (Girik's claw, 4 Nov), above it means a higher angle closes it. Never close further than this.
// PROVISIONAL (4 Nov): 3. With foam pads the empty jaws are still wider than a 19 mm die at 20, so the limit goes
// almost to the end; not 0, because 0 is the servo's own end stop and may stall even with the jaws empty. Girik is
// measuring the pad-to-pad gap at 90 and 20 to predict whether this is enough. Change it in BOTH files.
#ifndef GRIP_CLOSED_LIMIT
#define GRIP_CLOSED_LIMIT 3
#endif
#define GRIP_CLOSE_DIR (GRIP_CLOSED_LIMIT < GRIP_OPEN ? -1 : 1)  // -1: a lower angle closes
#define MIN_MS_PER_DEGREE 10       // speed limit: a move of 20 degrees takes at least 200 ms
#define MAX_GRIP_HOLD_MS 5000      // a closed grip relaxes after this long
#define MAX_LINE 64                // longer command lines are rejected

const int PINS[4] = {16, 17, 2, 4};          // base, shoulder, elbow, grip (kit sample code, untested)
const int MIN_ANGLE[3] = {0, 80, 30};        // base, shoulder, elbow
const int MAX_ANGLE[3] = {180, 180, 120};    // base, shoulder, elbow
const int HOME[4] = {90, 90, 90, 90};        // base, shoulder, elbow, grip
const int GRIP_RANGE[2] = {0, 140};          // the grip servo's whole range (4 Nov; 0 is its own end stop); the grip
                                             // is only ever sent from GRIP_CLOSED_LIMIT to the far end on the open
                                             // side (clamp_joint), so never to 0

#define GRIP_CLOSED_THRESHOLD 5    // the grip counts as closed once it is this far from open
#define HOME_MS 1500               // how long H takes (at least; the speed limit can make it longer)
#define RELAX_MS 500               // how long relaxing the grip takes (at least)
#define ATTACH_GAP_MS 400          // after E the servos are switched on one at a time, this far apart

// ---------- why the board last started (the B command) ----------
// The ESP32's own reset codes (esp_reset_reason(), ESP-IDF 4.4 in Arduino core 2.0.12). Plain numbers here so g++
// can test the mapping; mimic_arm.ino checks every number against the ESP32's own name when it compiles.
// These are ESP32 resets only: a sag in the servo battery that does not reset the ESP32 shows nothing here.
#define RESET_POWERON 1     // power-on; on the ESP32 also the EN (reset) button and an upload
#define RESET_SW 3          // software reset (esp_restart)
#define RESET_INT_WDT 5     // interrupt watchdog
#define RESET_TASK_WDT 6    // task watchdog
#define RESET_WDT 7         // other watchdogs
#define RESET_BROWNOUT 9    // the ESP32's supply voltage dropped too low

struct ArmState {
  float now[4];    // where each joint is right now, in degrees
  float from[4];   // where the current move started
  float to[4];     // where the current move ends
  unsigned long start_ms;
  unsigned long ms;
  bool moving;
  bool grip_closed;
  unsigned long grip_closed_since;
  int boot_code;   // why the board last started (esp_reset_reason()), set by mimic_arm.ino at start-up
  bool enabled;    // E received and no D since: the servos are being, or have been, switched on
  unsigned long enabled_ms;  // when E was received
  int attached;    // how many servos should be switched on now (base first, claw last); mimic_arm.ino follows it
};

inline const char *boot_reason_name(int code) {
  switch (code) {
    case RESET_POWERON: return "power-on";
    case RESET_BROWNOUT: return "brownout";
    case RESET_SW: return "software reset";
    case RESET_INT_WDT:
    case RESET_TASK_WDT:
    case RESET_WDT: return "watchdog";
    default: return "other";  // unknown, a crash (panic), waking from deep sleep, ...
  }
}

inline bool limits_ok() { return !LIMITS_ARE_PLACEHOLDER && LIMITS_CHECKED_BY_HAND; }

// While locked, the servos are never attached and no angle is ever written to them: they get no signal
// at all and stay limp wherever they are. mimic_arm.ino asks this before touching any servo.
inline bool servos_enabled() { return limits_ok(); }

// Staggered switch-on: how many servos should be switched on (attached) this long after E.
// Base at once, then shoulder at 400 ms, elbow at 800 ms, claw at 1200 ms, so their first jumps do not
// all draw current at the same moment (a battery dip could reset the board). While locked: none, ever.
inline int servos_to_attach(unsigned long ms_since_enable) {
  if (!servos_enabled()) return 0;
  unsigned long due = ms_since_enable / ATTACH_GAP_MS + 1;
  return due > 4 ? 4 : (int)due;
}

// Disarmed at power-on: nothing is attached until E. Called every tick; D (or a restart) makes it 0.
inline void update_attached(ArmState &a, unsigned long now_ms) {
  a.attached = a.enabled ? servos_to_attach(now_ms - a.enabled_ms) : 0;
}

// J and H are only allowed once all four servos are switched on.
inline bool armed(const ArmState &a) { return a.enabled && a.attached == 4; }

// Keep an angle inside its joint's range. For the grip: never past the closed limit on the closing side, and never
// past the end of GRIP_RANGE on the open side.
inline int clamp_joint(int joint, int angle) {
  int lo, hi;
  if (joint < 3) {
    lo = MIN_ANGLE[joint];
    hi = MAX_ANGLE[joint];
  } else {
    lo = GRIP_CLOSE_DIR < 0 ? GRIP_CLOSED_LIMIT : GRIP_RANGE[0];
    hi = GRIP_CLOSE_DIR < 0 ? GRIP_RANGE[1] : GRIP_CLOSED_LIMIT;
  }
  if (angle < lo) return lo;
  if (angle > hi) return hi;
  return angle;
}

// At power-on nothing is switched on (disarmed). The numbers start at HOME, but no servo is driven until E.
inline void arm_init(ArmState &a) {
  for (int j = 0; j < 4; j++) a.now[j] = a.from[j] = a.to[j] = HOME[j];
  a.start_ms = 0;
  a.ms = 0;
  a.moving = false;
  a.grip_closed = false;
  a.grip_closed_since = 0;
  a.boot_code = 0;
  a.enabled = false;
  a.enabled_ms = 0;
  a.attached = 0;
}

// E [b s e g]: where each servo starts when it is switched on (clamped, like J). Default 90 for each.
inline bool parse_enable(const char *line, int angles[4]) {
  for (int j = 0; j < 4; j++) angles[j] = 90;
  if (strcmp(line, "E") == 0) return true;
  int used = 0;
  return sscanf(line, "E %d %d %d %d%n", &angles[0], &angles[1], &angles[2], &angles[3], &used) == 4 &&
         line[used] == '\0';
}

// Start a smooth move from wherever the arm is now, so a new move simply replaces one still running.
inline void start_move(ArmState &a, const int target[4], unsigned long ms, unsigned long now_ms) {
  float biggest = 0;
  for (int j = 0; j < 4; j++) {
    a.from[j] = a.now[j];
    a.to[j] = clamp_joint(j, target[j]);
    float change = fabsf(a.to[j] - a.from[j]);
    if (change > biggest) biggest = change;
  }
  unsigned long slowest_allowed = (unsigned long)(biggest * MIN_MS_PER_DEGREE + 0.5f);
  a.ms = ms > slowest_allowed ? ms : slowest_allowed;
  a.start_ms = now_ms;
  a.moving = true;
}

inline void stop_arm(ArmState &a) {
  for (int j = 0; j < 4; j++) a.to[j] = a.now[j];
  a.moving = false;
}

// 0 -> 0, 1 -> 1, starting slow and ending slow, never going past 1.
inline float ease(float t) { return t * t * (3 - 2 * t); }

// Called every 20 ms. Moves the joints along the ramp; with no move running it changes nothing
// (the arm holds where it is and never repeats the last move).
inline void tick(ArmState &a, unsigned long now_ms) {
  if (a.moving) {
    float t = a.ms == 0 ? 1.0f : (float)(now_ms - a.start_ms) / (float)a.ms;
    if (t >= 1) {
      for (int j = 0; j < 4; j++) a.now[j] = a.to[j];
      a.moving = false;
    } else {
      float s = ease(t);
      for (int j = 0; j < 4; j++) a.now[j] = a.from[j] + (a.to[j] - a.from[j]) * s;
    }
  }

  // Overheating guard: a grip held closed for MAX_GRIP_HOLD_MS is relaxed to open.
  bool closed = fabsf(a.now[3] - GRIP_OPEN) > GRIP_CLOSED_THRESHOLD;
  if (closed && !a.grip_closed) {
    a.grip_closed = true;
    a.grip_closed_since = now_ms;
  }
  if (!closed) a.grip_closed = false;
  if (a.grip_closed && now_ms - a.grip_closed_since >= MAX_GRIP_HOLD_MS && a.to[3] != GRIP_OPEN) {
    int target[4] = {(int)lroundf(a.to[0]), (int)lroundf(a.to[1]), (int)lroundf(a.to[2]), GRIP_OPEN};
    start_move(a, target, RELAX_MS, now_ms);
  }
}

// One command line in, one reply line out (docs/plan.md 2.2).
inline void handle_line(ArmState &a, const char *line, unsigned long now_ms, char *reply, size_t n) {
  if (strlen(line) > MAX_LINE) {
    snprintf(reply, n, "ERR line too long");
    return;
  }
  if (strcmp(line, "?") == 0) {
    // " L" while locked, " D" while disarmed (or still switching on): then these are the firmware's own numbers,
    // not a real pose. No suffix: armed, and the servos are at these angles.
    const char *suffix = !servos_enabled() ? " L" : (armed(a) ? "" : " D");
    snprintf(reply, n, "A %d %d %d %d%s", (int)lroundf(a.now[0]), (int)lroundf(a.now[1]),
             (int)lroundf(a.now[2]), (int)lroundf(a.now[3]), suffix);
    return;
  }
  if (strcmp(line, "D") == 0) {  // always allowed: stop, then switch every servo off (limp)
    stop_arm(a);
    a.enabled = false;
    a.attached = 0;
    snprintf(reply, n, "OK");
    return;
  }
  if (line[0] == 'E') {
    int angles[4];
    if (!parse_enable(line, angles)) {
      snprintf(reply, n, "ERR bad command");
      return;
    }
    if (!limits_ok()) {
      snprintf(reply, n, "ERR limits not measured");
      return;
    }
    if (a.enabled) {  // starting again would reset the angles mid-move, and the servos would jump
      snprintf(reply, n, "ERR already enabled");
      return;
    }
    for (int j = 0; j < 4; j++) a.now[j] = a.from[j] = a.to[j] = clamp_joint(j, angles[j]);
    a.moving = false;
    a.grip_closed = false;
    a.enabled = true;
    a.enabled_ms = now_ms;
    update_attached(a, now_ms);  // the base at once; the others follow, 400 ms apart
    snprintf(reply, n, "OK");
    return;
  }
  if (strcmp(line, "B") == 0) {  // why the board last started, and how many seconds ago; read-only, always allowed
    snprintf(reply, n, "B %s %lu", boot_reason_name(a.boot_code), now_ms / 1000);
    return;
  }
  if (strcmp(line, "S") == 0) {  // always allowed: stopping must always work
    stop_arm(a);
    snprintf(reply, n, "OK");
    return;
  }
  if (strcmp(line, "H") == 0) {
    if (!limits_ok()) {
      snprintf(reply, n, "ERR limits not measured");
      return;
    }
    if (!armed(a)) {
      snprintf(reply, n, "ERR not enabled");
      return;
    }
    start_move(a, HOME, HOME_MS, now_ms);
    snprintf(reply, n, "OK");
    return;
  }
  if (line[0] == 'J') {
    int b, s, e, g, used = 0;
    long ms;
    if (sscanf(line, "J %d %d %d %d %ld%n", &b, &s, &e, &g, &ms, &used) == 5 && line[used] == '\0' && ms >= 0) {
      if (!limits_ok()) {
        snprintf(reply, n, "ERR limits not measured");
        return;
      }
      if (!armed(a)) {
        snprintf(reply, n, "ERR not enabled");
        return;
      }
      int target[4] = {b, s, e, g};
      start_move(a, target, (unsigned long)ms, now_ms);  // OK as soon as the move starts, not when it ends
      snprintf(reply, n, "OK");
      return;
    }
  }
  snprintf(reply, n, "ERR bad command");
}

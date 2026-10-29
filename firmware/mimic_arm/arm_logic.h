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
// PLACEHOLDER numbers until the arm is measured (needs.md B4).
// Moves are refused unless LIMITS_ARE_PLACEHOLDER is 0 AND LIMITS_CHECKED_BY_HAND is 1.
#ifndef LIMITS_ARE_PLACEHOLDER
#define LIMITS_ARE_PLACEHOLDER 1
#endif
#ifndef LIMITS_CHECKED_BY_HAND
#define LIMITS_CHECKED_BY_HAND 0
#endif
#define GRIP_OPEN 90
#define GRIP_CLOSED_LIMIT 120      // never close further than this (the grip servo overheats)
#define MIN_MS_PER_DEGREE 10       // speed limit: a move of 20 degrees takes at least 200 ms
#define MAX_GRIP_HOLD_MS 5000      // a closed grip relaxes after this long
#define MAX_LINE 64                // longer command lines are rejected

const int PINS[4] = {4, 5, 18, 19};          // base, shoulder, elbow, grip
const int MIN_ANGLE[3] = {70, 70, 70};       // base, shoulder, elbow
const int MAX_ANGLE[3] = {110, 110, 110};    // base, shoulder, elbow
const int HOME[4] = {90, 90, 90, 90};        // base, shoulder, elbow, grip

#define GRIP_CLOSED_THRESHOLD 5    // the grip counts as closed once it is this far from open
#define HOME_MS 1500               // how long H takes (at least; the speed limit can make it longer)
#define RELAX_MS 500               // how long relaxing the grip takes (at least)

struct ArmState {
  float now[4];    // where each joint is right now, in degrees
  float from[4];   // where the current move started
  float to[4];     // where the current move ends
  unsigned long start_ms;
  unsigned long ms;
  bool moving;
  bool grip_closed;
  unsigned long grip_closed_since;
};

inline bool limits_ok() { return !LIMITS_ARE_PLACEHOLDER && LIMITS_CHECKED_BY_HAND; }

// Keep an angle inside its joint's range. For the grip, the range is open .. closed limit.
inline int clamp_joint(int joint, int angle) {
  int lo, hi;
  if (joint < 3) {
    lo = MIN_ANGLE[joint];
    hi = MAX_ANGLE[joint];
  } else {
    lo = GRIP_OPEN < GRIP_CLOSED_LIMIT ? GRIP_OPEN : GRIP_CLOSED_LIMIT;
    hi = GRIP_OPEN < GRIP_CLOSED_LIMIT ? GRIP_CLOSED_LIMIT : GRIP_OPEN;
  }
  if (angle < lo) return lo;
  if (angle > hi) return hi;
  return angle;
}

// At power-on the servos are sent to HOME: a hobby servo cannot tell us where it is.
inline void arm_init(ArmState &a) {
  for (int j = 0; j < 4; j++) a.now[j] = a.from[j] = a.to[j] = HOME[j];
  a.start_ms = 0;
  a.ms = 0;
  a.moving = false;
  a.grip_closed = false;
  a.grip_closed_since = 0;
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
    snprintf(reply, n, "A %d %d %d %d", (int)lroundf(a.now[0]), (int)lroundf(a.now[1]),
             (int)lroundf(a.now[2]), (int)lroundf(a.now[3]));
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
      int target[4] = {b, s, e, g};
      start_move(a, target, (unsigned long)ms, now_ms);  // OK as soon as the move starts, not when it ends
      snprintf(reply, n, "OK");
      return;
    }
  }
  snprintf(reply, n, "ERR bad command");
}

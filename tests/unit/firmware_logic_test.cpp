// Tests for firmware/mimic_arm/arm_logic.h, compiled with g++ on the laptop by test_firmware_logic.py.
// Built three times: with the placeholder limits as they are, with only one of the two flags set, and
// with both flags set (-DLIMITS_ARE_PLACEHOLDER=0 -DLIMITS_CHECKED_BY_HAND=1) to test real moves.
// Angles come from the limits themselves, so these tests still make sense once the real numbers are in.
#include "arm_logic.h"

static int checks = 0, failures = 0;

#define CHECK(cond, name)                                    \
  do {                                                       \
    checks++;                                                \
    if (cond) {                                              \
      printf("ok   %s\n", name);                             \
    } else {                                                 \
      failures++;                                            \
      printf("FAIL %s  (line %d)\n", name, __LINE__);        \
    }                                                        \
  } while (0)

static bool near(float a, float b) { return fabsf(a - b) < 0.01f; }

static const char *send(ArmState &a, const char *line, unsigned long now_ms) {
  static char reply[64];
  handle_line(a, line, now_ms, reply, sizeof reply);
  return reply;
}

static void run_until(ArmState &a, unsigned long from_ms, unsigned long to_ms) {
  for (unsigned long t = from_ms; t <= to_ms; t += 20) tick(a, t);
}

static const int GRIP_DIR = GRIP_CLOSED_LIMIT > GRIP_OPEN ? 1 : -1;  // which way "closing" turns the servo

// ---------- always true, whatever the limits flags ----------

static void test_always() {
  ArmState a;
  arm_init(a);
  char expect[32];
  snprintf(expect, sizeof expect, "A %d %d %d %d", HOME[0], HOME[1], HOME[2], HOME[3]);
  CHECK(strcmp(send(a, "?", 0), expect) == 0, "? reports the home pose after power-on");
  CHECK(strcmp(send(a, "S", 0), "OK") == 0, "S always works");

  char long_line[80];
  memset(long_line, '1', 70);
  long_line[0] = 'J';
  long_line[70] = '\0';
  CHECK(strcmp(send(a, long_line, 0), "ERR line too long") == 0, "a 70-character line is rejected");
  char exactly_64[65];
  memset(exactly_64, ' ', 64);
  exactly_64[0] = 'X';
  exactly_64[64] = '\0';
  CHECK(strcmp(send(a, exactly_64, 0), "ERR bad command") == 0, "a 64-character line is read (then judged)");

  CHECK(strcmp(send(a, "X", 0), "ERR bad command") == 0, "unknown command rejected");
  CHECK(strcmp(send(a, "J 90 90 90", 0), "ERR bad command") == 0, "J needs 5 numbers");
  CHECK(strcmp(send(a, "J 90 90 90 90 800 7", 0), "ERR bad command") == 0, "J with extra numbers rejected");
  CHECK(strcmp(send(a, "J 90 90 90 90 -5", 0), "ERR bad command") == 0, "J with negative time rejected");
  CHECK(strcmp(send(a, "J 90 90 ninety 90 800", 0), "ERR bad command") == 0, "J with a word rejected");
}

// ---------- limits not ready: nothing may move ----------

static void test_refused() {
  ArmState a;
  arm_init(a);
  CHECK(!limits_ok(), "limits are not ok with these flags");
  CHECK(strcmp(send(a, "J 95 95 95 95 800", 0), "ERR limits not measured") == 0, "J refused");
  CHECK(strcmp(send(a, "H", 0), "ERR limits not measured") == 0, "H refused");
  run_until(a, 0, 5000);
  CHECK(!a.moving && near(a.now[0], HOME[0]) && near(a.now[3], HOME[3]), "the arm did not move");
}

// ---------- limits ready: moves, clamps, ramps, timers ----------

static void test_clamps() {
  CHECK(clamp_joint(0, MIN_ANGLE[0] - 50) == MIN_ANGLE[0], "base below its range is clamped to the minimum");
  CHECK(clamp_joint(0, MAX_ANGLE[0] + 50) == MAX_ANGLE[0], "base above its range is clamped to the maximum");
  CHECK(clamp_joint(1, MAX_ANGLE[1] + 1) == MAX_ANGLE[1], "shoulder 1 degree too far is clamped");
  CHECK(clamp_joint(2, MIN_ANGLE[2] - 1) == MIN_ANGLE[2], "elbow 1 degree too far is clamped");
  int inside = (MIN_ANGLE[1] + MAX_ANGLE[1]) / 2;
  CHECK(clamp_joint(1, inside) == inside, "an angle inside the range is unchanged");
  CHECK(clamp_joint(3, GRIP_CLOSED_LIMIT + 30 * GRIP_DIR) == GRIP_CLOSED_LIMIT, "grip never closes past its limit");
  CHECK(clamp_joint(3, GRIP_OPEN - 30 * GRIP_DIR) == GRIP_OPEN, "grip never opens past open");

  ArmState a;
  arm_init(a);
  char line[64];
  snprintf(line, sizeof line, "J %d %d %d %d 800", MIN_ANGLE[0] - 40, MAX_ANGLE[1] + 40, inside,
           GRIP_CLOSED_LIMIT + 40 * GRIP_DIR);
  CHECK(strcmp(send(a, line, 0), "OK") == 0, "an out-of-range J is accepted (and clamped)");
  CHECK(near(a.to[0], MIN_ANGLE[0]) && near(a.to[1], MAX_ANGLE[1]) && near(a.to[2], inside) &&
            near(a.to[3], GRIP_CLOSED_LIMIT),
        "J targets are clamped to the limits");
}

static void test_ramp() {
  ArmState a;
  arm_init(a);
  int target = HOME[0] + 10 <= MAX_ANGLE[0] ? HOME[0] + 10 : HOME[0] - 10;
  char line[64];
  snprintf(line, sizeof line, "J %d %d %d %d 1000", target, HOME[1], HOME[2], HOME[3]);
  CHECK(strcmp(send(a, line, 0), "OK") == 0, "J replies OK as soon as the move starts");
  CHECK(a.moving && near(a.now[0], HOME[0]), "nothing jumps when the move starts");

  float before = a.now[0], at_half_time = 0;
  bool steady = true, inside = true;
  for (unsigned long t = 20; t < 1000; t += 20) {
    tick(a, t);
    if (t == 500) at_half_time = a.now[0];
    float step = (a.now[0] - before) * (target > HOME[0] ? 1 : -1);
    if (step < -0.001f) steady = false;
    float lo = target < HOME[0] ? target : HOME[0], hi = target < HOME[0] ? HOME[0] : target;
    if (a.now[0] < lo - 0.001f || a.now[0] > hi + 0.001f) inside = false;
    before = a.now[0];
  }
  CHECK(steady, "the ramp only ever moves towards the target");
  CHECK(inside, "the ramp never overshoots");
  CHECK(near(at_half_time, (HOME[0] + target) / 2.0f), "halfway through the time, halfway there");
  tick(a, 1000);
  CHECK(near(a.now[0], target) && !a.moving, "the ramp ends exactly on the target");

  run_until(a, 1020, 60000);
  CHECK(near(a.now[0], target) && !a.moving, "with no new command the arm holds (no repeat, no drift)");
}

static void test_speed_limit() {
  ArmState a;
  arm_init(a);
  int target = MIN_ANGLE[0];
  char line[64];
  snprintf(line, sizeof line, "J %d %d %d %d 0", target, HOME[1], HOME[2], HOME[3]);
  send(a, line, 0);
  unsigned long needed = (unsigned long)(fabsf((float)(HOME[0] - target)) * MIN_MS_PER_DEGREE + 0.5f);
  CHECK(a.ms >= needed, "a too-fast move is slowed to the speed limit");
}

static void test_new_move_replaces_old() {
  ArmState a;
  arm_init(a);
  char line[64];
  snprintf(line, sizeof line, "J %d %d %d %d 1000", MAX_ANGLE[0], HOME[1], HOME[2], HOME[3]);
  send(a, line, 0);
  run_until(a, 20, 500);
  float midway = a.now[0];
  snprintf(line, sizeof line, "J %d %d %d %d 1000", MIN_ANGLE[0], HOME[1], HOME[2], HOME[3]);
  CHECK(strcmp(send(a, line, 500), "OK") == 0, "a new J during a move is accepted");
  CHECK(near(a.from[0], midway) && near(a.now[0], midway), "the new move starts from where the arm is (no jump)");
  run_until(a, 520, 5000);
  CHECK(near(a.now[0], MIN_ANGLE[0]), "the new move's target wins");
}

static void test_stop_and_home() {
  ArmState a;
  arm_init(a);
  char line[64];
  snprintf(line, sizeof line, "J %d %d %d %d 1000", MAX_ANGLE[0], HOME[1], HOME[2], HOME[3]);
  send(a, line, 0);
  run_until(a, 20, 400);
  float stopped_at = a.now[0];
  CHECK(strcmp(send(a, "S", 400), "OK") == 0, "S during a move replies OK");
  run_until(a, 420, 5000);
  CHECK(near(a.now[0], stopped_at) && !a.moving, "S stops and holds where the arm is");
  CHECK(strcmp(send(a, "H", 5000), "OK") == 0, "H accepted");
  run_until(a, 5020, 12000);
  CHECK(near(a.now[0], HOME[0]), "H goes back home");
}

static void test_grip_relaxes() {
  ArmState a;
  arm_init(a);
  char line[64];
  snprintf(line, sizeof line, "J %d %d %d %d 500", HOME[0], HOME[1], HOME[2], GRIP_CLOSED_LIMIT);
  send(a, line, 0);
  run_until(a, 20, 600);
  CHECK(near(a.now[3], GRIP_CLOSED_LIMIT), "the grip closes");
  unsigned long closed_at = a.grip_closed_since;
  run_until(a, 620, closed_at + MAX_GRIP_HOLD_MS - 20);
  CHECK(near(a.now[3], GRIP_CLOSED_LIMIT), "the grip stays closed until the hold time is up");
  run_until(a, closed_at + MAX_GRIP_HOLD_MS, closed_at + MAX_GRIP_HOLD_MS + 2000);
  CHECK(near(a.now[3], GRIP_OPEN), "after the hold time the grip relaxes to open");
  CHECK(near(a.now[0], HOME[0]) && near(a.now[1], HOME[1]), "relaxing the grip moves nothing else");

  ArmState b;
  arm_init(b);
  run_until(b, 0, MAX_GRIP_HOLD_MS * 3);
  CHECK(near(b.now[3], GRIP_OPEN) && !b.moving, "an open grip is left alone");
}

int main() {
  printf("limits flags: placeholder=%d checked_by_hand=%d\n", LIMITS_ARE_PLACEHOLDER, LIMITS_CHECKED_BY_HAND);
  test_always();
  if (limits_ok()) {
    test_clamps();
    test_ramp();
    test_speed_limit();
    test_new_move_replaces_old();
    test_stop_and_home();
    test_grip_relaxes();
  } else {
    test_refused();
  }
  printf("%d checks, %d failed\n", checks, failures);
  return failures ? 1 : 0;
}

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

// An unlocked arm after E, with all four servos switched on (E's default: 90 for every joint, which is HOME).
static ArmState enabled_arm() {
  ArmState a;
  arm_init(a);
  send(a, "E", 0);
  update_attached(a, 3 * ATTACH_GAP_MS);
  return a;
}

// ---------- always true, whatever the limits flags ----------

static void test_always() {
  ArmState a;
  arm_init(a);
  char expect[32];
  snprintf(expect, sizeof expect, "A %d %d %d %d%s", HOME[0], HOME[1], HOME[2], HOME[3], limits_ok() ? " D" : " L");
  CHECK(strcmp(send(a, "?", 0), expect) == 0, "? after power-on: disarmed (D), or L while locked");
  CHECK(!a.enabled && a.attached == 0, "power-on: disarmed, nothing attached");
  bool nothing = true;
  for (unsigned long t = 0; t <= 10000; t += 100) {
    update_attached(a, t);
    nothing = nothing && a.attached == 0;
  }
  CHECK(nothing, "power-on: without E no servo is ever switched on");
  CHECK(strcmp(send(a, "D", 0), "OK") == 0, "D always works (even disarmed or locked)");
  CHECK(strcmp(send(a, "E 90 90", 0), "ERR bad command") == 0, "E needs 0 or 4 numbers");
  CHECK(strcmp(send(a, "E 90 90 90 90 7", 0), "ERR bad command") == 0, "E with extra numbers rejected");
  CHECK(strcmp(send(a, "EX", 0), "ERR bad command") == 0, "EX is not E");
  CHECK(!a.enabled, "a bad E enables nothing");
  // grip: the closed limit sets the direction (GRIP_CLOSE_DIR); open and the limit are both inside the range
  CHECK(GRIP_RANGE[0] < GRIP_RANGE[1], "grip range is low-high");
  CHECK(GRIP_RANGE[0] <= GRIP_OPEN && GRIP_OPEN <= GRIP_RANGE[1], "grip open is inside the grip range");
  CHECK(GRIP_RANGE[0] <= GRIP_CLOSED_LIMIT && GRIP_CLOSED_LIMIT <= GRIP_RANGE[1], "grip closed limit is inside the range");
  CHECK(GRIP_CLOSE_DIR == (GRIP_CLOSED_LIMIT < GRIP_OPEN ? -1 : 1) && GRIP_CLOSED_LIMIT != GRIP_OPEN,
        "the closing direction comes from which side of open the closed limit is on");
  CHECK(clamp_joint(3, HOME[3]) == HOME[3], "grip home is an allowed angle");
  CHECK(servos_enabled() == limits_ok(), "servos are enabled exactly when the limits are ok");
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

  // B: why the board last started, in every build (locked or not), and it never moves anything.
  const char *expected_names[11] = {"other", "power-on", "other", "software reset", "other", "watchdog",
                                    "watchdog", "watchdog", "other", "brownout", "other"};
  bool all_named = true;
  for (int code = 0; code <= 10; code++) all_named = all_named && strcmp(boot_reason_name(code), expected_names[code]) == 0;
  CHECK(all_named, "reset codes 0-10 are named: 1 power-on, 3 software reset, 5-7 watchdog, 9 brownout, rest other");
  CHECK(strcmp(boot_reason_name(99), "other") == 0, "an unknown reset code is other");
  ArmState b;
  arm_init(b);
  CHECK(strcmp(send(b, "B", 500), "B other 0") == 0, "before setup sets the code, B says other");
  b.boot_code = RESET_BROWNOUT;
  CHECK(strcmp(send(b, "B", 42999), "B brownout 42") == 0, "B gives the reason and whole seconds since start-up");
  b.boot_code = RESET_SW;
  CHECK(strcmp(send(b, "B", 7000), "B software reset 7") == 0, "a two-word reason");
  run_until(b, 0, 2000);
  CHECK(!b.moving && near(b.now[0], HOME[0]) && near(b.now[3], HOME[3]), "B moves nothing");
  CHECK(strcmp(send(b, "B 1", 0), "ERR bad command") == 0, "B takes no arguments");
  CHECK(strcmp(send(b, "b", 0), "ERR bad command") == 0, "commands are case-sensitive: b is not B");

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
  CHECK(!servos_enabled(), "locked: servos are not enabled (never attached, no angle written)");
  bool none = true;
  for (unsigned long t = 0; t <= 10000; t += 100) none = none && servos_to_attach(t) == 0;
  CHECK(none, "locked: the staggered start-up never attaches a servo");
  CHECK(strstr(send(a, "?", 0), " L") != NULL, "locked: ? ends with L");
  CHECK(strcmp(send(a, "J 95 95 95 95 800", 0), "ERR limits not measured") == 0, "J refused");
  CHECK(strcmp(send(a, "H", 0), "ERR limits not measured") == 0, "H refused");
  CHECK(strcmp(send(a, "E", 0), "ERR limits not measured") == 0, "locked: E refused");
  CHECK(strcmp(send(a, "E 90 90 90 90", 0), "ERR limits not measured") == 0, "locked: E with angles refused");
  update_attached(a, 5000);
  CHECK(!a.enabled && a.attached == 0, "locked: E switches nothing on");
  run_until(a, 0, 5000);
  CHECK(!a.moving && near(a.now[0], HOME[0]) && near(a.now[3], HOME[3]), "the arm did not move");
}

// ---------- limits ready: moves, clamps, ramps, timers ----------

static void test_staggered_start() {
  CHECK(servos_to_attach(0) == 1, "switch-on: the base is switched on at once");
  CHECK(servos_to_attach(ATTACH_GAP_MS - 1) == 1, "switch-on: still only the base just before 400 ms");
  CHECK(servos_to_attach(ATTACH_GAP_MS) == 2, "switch-on: the shoulder at 400 ms");
  CHECK(servos_to_attach(2 * ATTACH_GAP_MS) == 3, "switch-on: the elbow at 800 ms");
  CHECK(servos_to_attach(3 * ATTACH_GAP_MS) == 4, "switch-on: the claw at 1200 ms");
  CHECK(servos_to_attach(60000) == 4, "switch-on: never more than 4");
}

static void test_ramp_speed() {
  // Even a J sent with 0 ms is slowed to MIN_MS_PER_DEGREE, and the ease-in/ease-out peak is 1.5x the average
  // speed, so no 20 ms tick may move a joint more than 1.5 * 20 / MIN_MS_PER_DEGREE degrees.
  ArmState a = enabled_arm();
  char line[64];
  snprintf(line, sizeof line, "J %d %d %d %d 0", MIN_ANGLE[0], MAX_ANGLE[1], MIN_ANGLE[2], HOME[3]);
  send(a, line, 0);
  float worst = 0, before[4] = {a.now[0], a.now[1], a.now[2], a.now[3]};
  for (unsigned long t = 20; t <= 5000; t += 20) {
    tick(a, t);
    for (int j = 0; j < 4; j++) {
      float step = fabsf(a.now[j] - before[j]);
      if (step > worst) worst = step;
      before[j] = a.now[j];
    }
  }
  CHECK(worst <= 1.5f * 20 / MIN_MS_PER_DEGREE + 0.01f, "ramp: no 20 ms tick moves a joint more than 3 degrees");
  CHECK(near(a.now[0], MIN_ANGLE[0]) && !a.moving, "ramp: the fast J still arrives");
}

static void test_unlocked_status() {
  ArmState a;
  arm_init(a);
  CHECK(servos_enabled(), "unlocked: servos are enabled");
  CHECK(strchr(send(a, "?", 0), 'L') == NULL, "unlocked: ? has no L");
  ArmState b = enabled_arm();
  char expect[32];
  snprintf(expect, sizeof expect, "A %d %d %d %d", HOME[0], HOME[1], HOME[2], HOME[3]);
  CHECK(strcmp(send(b, "?", 0), expect) == 0, "armed: ? has no suffix");
}

static void test_disarmed_until_E() {
  ArmState a;
  arm_init(a);
  CHECK(strcmp(send(a, "J 95 95 95 95 800", 0), "ERR not enabled") == 0, "disarmed: J refused");
  CHECK(strcmp(send(a, "H", 0), "ERR not enabled") == 0, "disarmed: H refused");
  CHECK(strcmp(send(a, "S", 0), "OK") == 0, "disarmed: S works");
  run_until(a, 0, 3000);
  CHECK(!a.moving && near(a.now[0], HOME[0]), "disarmed: nothing moves");
}

static void test_enable_switches_on_one_at_a_time() {
  ArmState a;
  arm_init(a);
  const unsigned long t0 = 10000;  // E at 10 s after power-on: the times count from E
  CHECK(strcmp(send(a, "E", t0), "OK") == 0, "E replies OK");
  CHECK(a.attached == 1, "E: the base at once");
  update_attached(a, t0 + ATTACH_GAP_MS - 1);
  CHECK(a.attached == 1, "E: still only the base just before 400 ms");
  CHECK(strcmp(send(a, "J 95 95 95 95 800", t0 + 100), "ERR not enabled") == 0, "J refused while switching on");
  CHECK(strstr(send(a, "?", t0 + 100), " D") != NULL, "? still ends with D while switching on");
  update_attached(a, t0 + ATTACH_GAP_MS);
  CHECK(a.attached == 2, "E: the shoulder at 400 ms");
  update_attached(a, t0 + 2 * ATTACH_GAP_MS);
  CHECK(a.attached == 3, "E: the elbow at 800 ms");
  update_attached(a, t0 + 3 * ATTACH_GAP_MS);
  CHECK(a.attached == 4 && armed(a), "E: the claw at 1200 ms, then armed");
  update_attached(a, t0 + 60000);
  CHECK(a.attached == 4, "E: never more than 4");
  CHECK(strcmp(send(a, "J 95 95 95 95 800", t0 + 1200), "OK") == 0, "armed: J accepted");
  CHECK(strcmp(send(a, "E", t0 + 1300), "ERR already enabled") == 0, "E twice is refused (no reset mid-move)");
  CHECK(a.moving, "a refused second E leaves the move running");
}

static void test_enable_starts_at_the_given_angles() {
  ArmState a;
  arm_init(a);
  CHECK(strcmp(send(a, "E 100 85 70 95", 0), "OK") == 0, "E with angles replies OK");
  CHECK(near(a.now[0], 100) && near(a.now[1], 85) && near(a.now[2], 70) && near(a.now[3], 95) && !a.moving,
        "E sets where each servo starts, and nothing moves");
  ArmState b;
  arm_init(b);
  char line[64];
  snprintf(line, sizeof line, "E %d %d %d %d", MIN_ANGLE[0] - 20, MAX_ANGLE[1] + 20, HOME[2], GRIP_RANGE[1] + 20);
  send(b, line, 0);
  CHECK(near(b.now[0], MIN_ANGLE[0]) && near(b.now[1], MAX_ANGLE[1]) && near(b.now[3], GRIP_RANGE[1]),
        "E angles are clamped like J");
}

static void test_disable_goes_limp() {
  ArmState a = enabled_arm();
  char line[64];
  snprintf(line, sizeof line, "J %d %d %d %d 1000", MAX_ANGLE[0], HOME[1], HOME[2], HOME[3]);
  send(a, line, 0);
  run_until(a, 20, 400);
  float stopped_at = a.now[0];
  CHECK(strcmp(send(a, "D", 400), "OK") == 0, "D replies OK");
  CHECK(!a.enabled && a.attached == 0 && !a.moving, "D: stopped, disarmed, nothing attached");
  update_attached(a, 5000);
  run_until(a, 420, 3000);
  CHECK(a.attached == 0 && near(a.now[0], stopped_at), "after D nothing is switched on and nothing moves");
  CHECK(strcmp(send(a, "J 95 95 95 95 800", 3000), "ERR not enabled") == 0, "after D, J refused");
  CHECK(strstr(send(a, "?", 3000), " D") != NULL, "after D, ? ends with D");
  CHECK(strcmp(send(a, "E", 3000), "OK") == 0, "E works again after D");
}

static void test_clamps() {
  CHECK(clamp_joint(0, MIN_ANGLE[0] - 50) == MIN_ANGLE[0], "base below its range is clamped to the minimum");
  CHECK(clamp_joint(0, MAX_ANGLE[0] + 50) == MAX_ANGLE[0], "base above its range is clamped to the maximum");
  CHECK(clamp_joint(1, MAX_ANGLE[1] + 1) == MAX_ANGLE[1], "shoulder 1 degree too far is clamped");
  CHECK(clamp_joint(2, MIN_ANGLE[2] - 1) == MIN_ANGLE[2], "elbow 1 degree too far is clamped");
  int inside = (MIN_ANGLE[1] + MAX_ANGLE[1]) / 2;
  CHECK(clamp_joint(1, inside) == inside, "an angle inside the range is unchanged");
  CHECK(clamp_joint(3, GRIP_CLOSED_LIMIT + 30 * GRIP_DIR) == GRIP_CLOSED_LIMIT, "grip never closes past its limit");
  CHECK(clamp_joint(3, GRIP_CLOSED_LIMIT + 1 * GRIP_DIR) == GRIP_CLOSED_LIMIT, "grip: 1 degree past the limit is clamped");
  CHECK(clamp_joint(3, GRIP_CLOSED_LIMIT) == GRIP_CLOSED_LIMIT, "grip can close right up to its limit");
  CHECK(clamp_joint(3, GRIP_OPEN + 5 * GRIP_DIR) == GRIP_OPEN + 5 * GRIP_DIR, "grip can close 5 degrees from open");
  int open_end = GRIP_DIR < 0 ? GRIP_RANGE[1] : GRIP_RANGE[0];  // the far end of the range, on the open side
  CHECK(clamp_joint(3, open_end) == open_end, "grip can open to the end of its range");
  CHECK(clamp_joint(3, open_end - 10 * GRIP_DIR) == open_end, "grip never opens past the end of its range");

  ArmState a = enabled_arm();
  char line[64];
  int elbow_inside = (MIN_ANGLE[2] + MAX_ANGLE[2]) / 2;  // each joint has its own range
  snprintf(line, sizeof line, "J %d %d %d %d 800", MIN_ANGLE[0] - 40, MAX_ANGLE[1] + 40, elbow_inside,
           GRIP_CLOSED_LIMIT + 40 * GRIP_DIR);
  CHECK(strcmp(send(a, line, 0), "OK") == 0, "an out-of-range J is accepted (and clamped)");
  CHECK(near(a.to[0], MIN_ANGLE[0]) && near(a.to[1], MAX_ANGLE[1]) && near(a.to[2], elbow_inside) &&
            near(a.to[3], GRIP_CLOSED_LIMIT),
        "J targets are clamped to the limits");
}

static void test_ramp() {
  ArmState a = enabled_arm();
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
  ArmState a = enabled_arm();
  int target = MIN_ANGLE[0];
  char line[64];
  snprintf(line, sizeof line, "J %d %d %d %d 0", target, HOME[1], HOME[2], HOME[3]);
  send(a, line, 0);
  unsigned long needed = (unsigned long)(fabsf((float)(HOME[0] - target)) * MIN_MS_PER_DEGREE + 0.5f);
  CHECK(a.ms >= needed, "a too-fast move is slowed to the speed limit");
}

static void test_new_move_replaces_old() {
  ArmState a = enabled_arm();
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
  ArmState a = enabled_arm();
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
  ArmState a = enabled_arm();
  char line[64];
  snprintf(line, sizeof line, "J %d %d %d %d 500", HOME[0], HOME[1], HOME[2], GRIP_CLOSED_LIMIT);
  send(a, line, 0);
  unsigned long arrive = a.ms;  // the move time, after the speed limit (90 -> 20 takes 700 ms)
  run_until(a, 20, arrive + 100);
  CHECK(near(a.now[3], GRIP_CLOSED_LIMIT), "the grip closes");
  unsigned long closed_at = a.grip_closed_since;
  run_until(a, arrive + 120, closed_at + MAX_GRIP_HOLD_MS - 20);
  CHECK(near(a.now[3], GRIP_CLOSED_LIMIT), "the grip stays closed until the hold time is up");
  run_until(a, closed_at + MAX_GRIP_HOLD_MS, closed_at + MAX_GRIP_HOLD_MS + 2000);
  CHECK(near(a.now[3], GRIP_OPEN), "after the hold time the grip relaxes to open");
  CHECK(near(a.now[0], HOME[0]) && near(a.now[1], HOME[1]), "relaxing the grip moves nothing else");

  ArmState b = enabled_arm();
  run_until(b, 0, MAX_GRIP_HOLD_MS * 3);
  CHECK(near(b.now[3], GRIP_OPEN) && !b.moving, "an open grip is left alone");
}

int main() {
  printf("limits flags: placeholder=%d checked_by_hand=%d; grip closes %s (closed limit %d)\n", LIMITS_ARE_PLACEHOLDER,
         LIMITS_CHECKED_BY_HAND, GRIP_CLOSE_DIR < 0 ? "downwards" : "upwards", GRIP_CLOSED_LIMIT);
  test_always();
  if (limits_ok()) {
    test_staggered_start();
    test_disarmed_until_E();
    test_enable_switches_on_one_at_a_time();
    test_enable_starts_at_the_given_angles();
    test_disable_goes_limp();
    test_ramp_speed();
    test_unlocked_status();
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

/* Connect Four solver (7 columns x 6 rows): bitboards, negamax with alpha-beta, a transposition table, center-first
 * move ordering and only non-losing moves, following the well-known approach of Pascal Pons' tutorial.
 *
 * A position is given as its moves, a string of column digits '1'..'7'. Scores are from the side to move: positive
 * means it wins (the larger, the sooner), 0 a draw, negative a loss. With weak = 1 only the sign is exact
 * (-1, 0 or 1), which is much faster.
 *
 *   int c4_solve(const char *moves, int weak)             score of the position (INVALID for a bad move string)
 *   int c4_analyze(const char *moves, int weak, int *out)  out[c] = score of playing column c (0-based) from the
 *                                                          mover's side, UNPLAYABLE for a full column; returns 0 or -1
 *
 * Build: cc -O3 -shared -fPIC -o libc4solver.so solver.c
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define W 7
#define H 6
#define MIN_SCORE (-(W * H) / 2 + 3)
#define MAX_SCORE ((W * H + 1) / 2 - 3)
#define INVALID -10000
#define UNPLAYABLE -1000

typedef uint64_t u64;

static const u64 BOTTOM = 0x0000040810204081ULL;          /* the lowest cell of every column (H + 1 bits per column) */
static const u64 BOARD = 0x0000040810204081ULL * 0x3fULL;  /* every playable cell */

typedef struct {
  u64 current; /* stones of the side to move */
  u64 mask;    /* all stones */
  int moves;
} Pos;

static u64 col_bottom(int c) { return 1ULL << (c * (H + 1)); }
static u64 col_top(int c) { return (1ULL << (H - 1)) << (c * (H + 1)); }
static u64 col_mask(int c) { return ((1ULL << H) - 1) << (c * (H + 1)); }

static int can_play(const Pos *p, int c) { return (p->mask & col_top(c)) == 0; }
static void play_move(Pos *p, u64 move) {
  p->current ^= p->mask;
  p->mask |= move;
  p->moves++;
}
static u64 move_of_col(const Pos *p, int c) { return (p->mask + col_bottom(c)) & col_mask(c); }

static u64 winning_cells(u64 pos, u64 mask) {
  u64 r = (pos << 1) & (pos << 2) & (pos << 3); /* vertical */
  u64 q;
  int s[3] = {H + 1, H, H + 2};                 /* horizontal and both diagonals */
  for (int k = 0; k < 3; k++) {
    int d = s[k];
    q = (pos << d) & (pos << 2 * d);
    r |= q & (pos << 3 * d);
    r |= q & (pos >> d);
    q = (pos >> d) & (pos >> 2 * d);
    r |= q & (pos << d);
    r |= q & (pos >> 3 * d);
  }
  return r & (BOARD ^ mask);
}
static u64 possible(const Pos *p) { return (p->mask + BOTTOM) & BOARD; }
static u64 my_wins(const Pos *p) { return winning_cells(p->current, p->mask); }
static u64 their_wins(const Pos *p) { return winning_cells(p->current ^ p->mask, p->mask); }
static int can_win_next(const Pos *p) { return (my_wins(p) & possible(p)) != 0; }
static int is_winning_col(const Pos *p, int c) { return (my_wins(p) & possible(p) & col_mask(c)) != 0; }

static u64 non_losing_moves(const Pos *p) {
  u64 moves = possible(p);
  u64 opp = their_wins(p);
  u64 forced = moves & opp;
  if (forced) {
    if (forced & (forced - 1)) return 0; /* two threats: every move loses */
    moves = forced;
  }
  return moves & ~(opp >> 1); /* never play right below an opponent's winning cell */
}

/* ---- transposition table: upper bounds, keyed by the position ------------------------------------ */
#define TT_SIZE 8388617 /* a prime */
static u64 *tt_key;
static int8_t *tt_val;
static int tt_ready(void) {
  if (!tt_key) {
    tt_key = calloc(TT_SIZE, sizeof(u64));
    tt_val = calloc(TT_SIZE, sizeof(int8_t));
  }
  return tt_key && tt_val;
}
static u64 key_of(const Pos *p) { return p->current + p->mask; }
static void tt_put(u64 k, int v) {
  size_t i = k % TT_SIZE;
  tt_key[i] = k;
  tt_val[i] = (int8_t)v;
}
static int tt_get(u64 k) {
  size_t i = k % TT_SIZE;
  return tt_key[i] == k ? tt_val[i] : 0;
}

static int popcount(u64 m) { return __builtin_popcountll(m); }

/* Precondition: nobody has won and the side to move cannot win with its next move. */
static int negamax(const Pos *p, int alpha, int beta) {
  u64 next = non_losing_moves(p);
  if (!next) return -(W * H - p->moves) / 2;
  if (p->moves >= W * H - 2) return 0;
  int lo = -(W * H - 2 - p->moves) / 2;
  if (alpha < lo) {
    alpha = lo;
    if (alpha >= beta) return alpha;
  }
  int hi = (W * H - 1 - p->moves) / 2;
  int stored = tt_get(key_of(p));
  if (stored) hi = stored + MIN_SCORE - 1;
  if (beta > hi) {
    beta = hi;
    if (alpha >= beta) return beta;
  }
  /* candidates in center-first order, then by how many winning cells they create (stable) */
  u64 mv[W];
  int sc[W], n = 0;
  for (int i = 0; i < W; i++) {
    int c = W / 2 + (1 - 2 * (i % 2)) * (i + 1) / 2;
    u64 m = next & col_mask(c);
    if (!m) continue;
    int score = popcount(winning_cells(p->current | m, p->mask));
    int j = n++;
    while (j > 0 && sc[j - 1] < score) {
      mv[j] = mv[j - 1];
      sc[j] = sc[j - 1];
      j--;
    }
    mv[j] = m;
    sc[j] = score;
  }
  for (int i = 0; i < n; i++) {
    Pos q = *p;
    play_move(&q, mv[i]);
    int s = -negamax(&q, -beta, -alpha);
    if (s >= beta) return s;
    if (s > alpha) alpha = s;
  }
  tt_put(key_of(p), alpha - MIN_SCORE + 1);
  return alpha;
}

static int solve_pos(const Pos *p, int weak) {
  if (can_win_next(p)) return (W * H + 1 - p->moves) / 2;
  int lo = -(W * H - p->moves) / 2, hi = (W * H + 1 - p->moves) / 2;
  if (weak) {
    lo = -1;
    hi = 1;
  }
  while (lo < hi) { /* null-window searches, probing near zero first */
    int med = lo + (hi - lo) / 2;
    if (med <= 0 && lo / 2 < med) med = lo / 2;
    else if (med >= 0 && hi / 2 > med) med = hi / 2;
    int r = negamax(p, med, med + 1);
    if (r <= med) hi = r;
    else lo = r;
  }
  return lo;
}

/* Replays a move string; -1 on a bad or winning-before-the-end sequence. */
static int load(Pos *p, const char *moves) {
  memset(p, 0, sizeof *p);
  for (const char *s = moves; *s; s++) {
    int c = *s - '1';
    if (c < 0 || c >= W || !can_play(p, c) || is_winning_col(p, c)) return -1;
    play_move(p, move_of_col(p, c));
  }
  return 0;
}

int c4_solve(const char *moves, int weak) {
  Pos p;
  if (!tt_ready() || load(&p, moves)) return INVALID;
  return solve_pos(&p, weak);
}

int c4_analyze(const char *moves, int weak, int *out) {
  Pos p;
  if (!tt_ready() || load(&p, moves)) return -1;
  for (int c = 0; c < W; c++) {
    if (!can_play(&p, c)) {
      out[c] = UNPLAYABLE;
    } else if (is_winning_col(&p, c)) {
      out[c] = (W * H + 1 - p.moves) / 2;
    } else {
      Pos q = p;
      play_move(&q, move_of_col(&q, c));
      out[c] = q.moves == W * H ? 0 : -solve_pos(&q, weak);
    }
  }
  return 0;
}

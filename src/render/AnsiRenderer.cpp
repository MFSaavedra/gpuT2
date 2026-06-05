/**
 * @file AnsiRenderer.cpp
 * @brief Implementation of the in-place ANSI terminal renderer, with optional
 *        arrow-key panning of the viewport over a grid larger than the screen.
 */

#include "gol/render/AnsiRenderer.hpp"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <string>
#include <thread>
#include <utility>

#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include "gol/Grid.hpp"

namespace gol {

namespace {

// ANSI CSI / DEC private-mode control sequences.
constexpr const char* kHideCursor  = "\033[?25l";   // hide the text cursor
constexpr const char* kShowCursor  = "\033[?25h";   // show the text cursor
constexpr const char* kWrapOff     = "\033[?7l";    // disable autowrap (wide rows clip)
constexpr const char* kWrapOn      = "\033[?7h";    // re-enable autowrap
constexpr const char* kClearScreen = "\033[2J";     // erase the whole screen
constexpr const char* kCursorHome  = "\033[H";      // move cursor to top-left
constexpr const char* kSyncBegin   = "\033[?2026h"; // atomic frame update (ignored if unsupported)
constexpr const char* kSyncEnd     = "\033[?2026l";
constexpr const char* kClearToEol  = "\033[K";      // erase from cursor to end of line
constexpr const char* kClearBelow  = "\033[J";      // erase from cursor to end of screen

// Saved terminal state so it can be restored from the destructor AND from a
// signal handler (only one AnsiRenderer exists at a time). On SIGINT/SIGTERM the
// default handler would leave the terminal in raw mode without these.
struct termios g_origTermios;
bool g_termiosSaved = false;

/// @brief Restore canonical mode + cursor + autowrap. Async-signal-safe.
void restoreTerminal() {
  if (g_termiosSaved) tcsetattr(STDIN_FILENO, TCSANOW, &g_origTermios);
  const char seq[] = "\033[?7h\033[?25h"; // wrap on, cursor on
  ssize_t r = write(STDOUT_FILENO, seq, sizeof(seq) - 1);
  (void)r;
}

/// @brief Signal handler: restore the terminal, then re-raise with the default.
void onSignal(int sig) {
  restoreTerminal();
  std::signal(sig, SIG_DFL);
  std::raise(sig);
}

/**
 * @brief Display width of a glyph in terminal columns.
 *
 * Counts UTF-8 code points (lead bytes), assuming one column each -- true for the
 * glyphs used here (ASCII, spaces, block elements). Double-width (East Asian
 * Wide) characters are not accounted for.
 * @param s Glyph string.
 * @return Number of terminal columns the glyph occupies.
 */
std::size_t displayWidth(const std::string& s) {
  std::size_t w = 0;
  for (unsigned char c : s)
    if ((c & 0xC0) != 0x80) ++w; // count bytes that are not UTF-8 continuation bytes
  return w;
}

/**
 * @brief Query the terminal size, falling back to 80x24 when stdout is not a TTY.
 * @param[out] rows Visible terminal rows.
 * @param[out] cols Visible terminal columns.
 */
void terminalSize(unsigned& rows, unsigned& cols) {
  struct winsize ws {};
  if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 0 && ws.ws_col > 0) {
    rows = ws.ws_row;
    cols = ws.ws_col;
  } else {
    rows = 24;
    cols = 80;
  }
}

/// @brief Saturating subtract for unsigned offsets (clamps at 0).
std::size_t subSat(std::size_t a, std::size_t b) { return a > b ? a - b : 0; }

} // namespace

AnsiRenderer::AnsiRenderer(std::string alive, std::string dead, unsigned delayMs)
    : alive_(std::move(alive)), dead_(std::move(dead)), delayMs_(delayMs) {
  interactive_ = isatty(STDIN_FILENO) != 0;
  if (interactive_) {
    // Raw-ish mode: no line buffering, no echo, non-blocking reads (VMIN=0,
    // VTIME=0 -> read() returns immediately with whatever is queued). Keep ISIG
    // so Ctrl-C still works; the signal handler restores the terminal.
    if (tcgetattr(STDIN_FILENO, &g_origTermios) == 0) {
      g_termiosSaved = true;
      struct termios raw = g_origTermios;
      raw.c_lflag &= ~(static_cast<tcflag_t>(ICANON | ECHO));
      raw.c_cc[VMIN] = 0;
      raw.c_cc[VTIME] = 0;
      tcsetattr(STDIN_FILENO, TCSANOW, &raw);
      std::signal(SIGINT, onSignal);
      std::signal(SIGTERM, onSignal);
    } else {
      interactive_ = false; // not a usable TTY after all
    }
  }
  std::fputs(kHideCursor, stdout);
  std::fputs(kWrapOff, stdout); // wide rows clip at the right edge instead of wrapping
  std::fputs(kClearScreen, stdout);
  std::fflush(stdout);
}

AnsiRenderer::~AnsiRenderer() {
  // Leave the terminal in a sane state and move the prompt below the last frame.
  std::fputs(kWrapOn, stdout);
  std::fputs(kShowCursor, stdout);
  std::fputs("\n", stdout);
  std::fflush(stdout);
  if (interactive_ && g_termiosSaved) {
    tcsetattr(STDIN_FILENO, TCSANOW, &g_origTermios);
    g_termiosSaved = false;
    std::signal(SIGINT, SIG_DFL);
    std::signal(SIGTERM, SIG_DFL);
  }
}

void AnsiRenderer::render(const Grid& grid, std::uint64_t generation) {
  // The body runs once normally. While paused (interactive only) it loops --
  // still polling input so you can pan and unpause -- redrawing the SAME
  // generation, and returns once unpaused or quitting. The main loop and engine
  // are untouched: pause is entirely contained here.
  do {
    unsigned termRows = 24;
    unsigned termCols = 80;
    terminalSize(termRows, termCols);

    // Each cell is `cellCols` terminal columns wide (the default "██" is 2). Clip
    // the board to the visible viewport (in cells).
    const std::size_t cellCols =
        std::max<std::size_t>(1, std::max(displayWidth(alive_), displayWidth(dead_)));
    const std::size_t visRows =
        std::min<std::size_t>(grid.rows(), termRows > 1 ? termRows - 1u : 1u);
    const std::size_t visCols = std::min<std::size_t>(grid.cols(), termCols / cellCols);
    const bool clipped = visRows < grid.rows() || visCols < grid.cols();

    // Drain queued key presses (non-blocking): arrows pan, space/p toggles pause,
    // q quits. Offsets are clamped below, so a held key cannot run off-grid.
    if (interactive_) {
      const std::size_t stepX = std::max<std::size_t>(1, visCols / 8);
      const std::size_t stepY = std::max<std::size_t>(1, visRows / 8);
      char buf[64];
      ssize_t n = read(STDIN_FILENO, buf, sizeof(buf));
      for (ssize_t i = 0; i < n; ++i) {
        const char c = buf[i];
        if (c == 'q' || c == 'Q') {
          quit_ = true;
        } else if (c == ' ' || c == 'p' || c == 'P') {
          paused_ = !paused_;
        } else if (c == '\033' && i + 2 < n && buf[i + 1] == '[') {
          switch (buf[i + 2]) {
            case 'A': offsetY_ = subSat(offsetY_, stepY); break; // up
            case 'B': offsetY_ += stepY; break;                  // down
            case 'C': offsetX_ += stepX; break;                  // right
            case 'D': offsetX_ = subSat(offsetX_, stepX); break; // left
            default: break;
          }
          i += 2;
        }
      }
    }

    // Clamp the viewport so it stays inside the grid.
    offsetX_ = std::min(offsetX_, subSat(grid.cols(), visCols));
    offsetY_ = std::min(offsetY_, subSat(grid.rows(), visRows));

    const std::size_t glyph = std::max(alive_.size(), dead_.size()); // bytes per cell
    std::string out;
    out.reserve(visRows * (visCols * glyph + 4) + 128);
    out += kSyncBegin;
    out += kCursorHome;

    out += "generation ";
    out += std::to_string(generation);
    if (paused_) out += "  [PAUSED]";
    if (clipped) {
      out += " view(";
      out += std::to_string(offsetX_);
      out += ",";
      out += std::to_string(offsetY_);
      out += ") ";
      out += std::to_string(visCols);
      out += "x";
      out += std::to_string(visRows);
      out += " of ";
      out += std::to_string(grid.cols());
      out += "x";
      out += std::to_string(grid.rows());
    }
    if (interactive_) out += "  [arrows pan, space pause, q quit]";
    out += kClearToEol;
    out += '\n';

    for (std::size_t y = 0; y < visRows; ++y) {
      for (std::size_t x = 0; x < visCols; ++x) {
        out += (grid.at(offsetX_ + x, offsetY_ + y) ? alive_ : dead_);
      }
      out += kClearToEol; // wipe any leftover from a previous wider frame
      out += '\n';
    }
    out += kClearBelow; // wipe leftover rows if the terminal shrank
    out += kSyncEnd;

    std::fputs(out.c_str(), stdout);
    std::fflush(stdout);

    if (delayMs_ > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(delayMs_));
    }
  } while (interactive_ && paused_ && !quit_);
}

} // namespace gol

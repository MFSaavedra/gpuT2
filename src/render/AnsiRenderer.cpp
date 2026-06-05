/**
 * @file AnsiRenderer.cpp
 * @brief Implementation of the in-place ANSI terminal renderer.
 */

#include "gol/render/AnsiRenderer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <utility>

#include <sys/ioctl.h>
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

/**
 * @brief Display width of a glyph in terminal columns.
 *
 * Counts UTF-8 code points (lead bytes), assuming one column each -- true for the
 * glyphs used here (ASCII, spaces, block elements). Double-width (East Asian
 * Wide) characters are not accounted for.
 * @param s Glyph string.
 * @return Number of terminal columns the glyph occupies (>= 0).
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

} // namespace

AnsiRenderer::AnsiRenderer(std::string alive, std::string dead, unsigned delayMs)
    : alive_(std::move(alive)), dead_(std::move(dead)), delayMs_(delayMs) {
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
}

void AnsiRenderer::render(const Grid& grid, std::uint64_t generation) {
  unsigned termRows = 24;
  unsigned termCols = 80;
  terminalSize(termRows, termCols);

  // Each cell may be more than one terminal column wide (the default glyph is
  // two full blocks, "██", so cells look square). Clip by DISPLAY width, not
  // bytes: how many whole cells fit across the terminal.
  const std::size_t cellCols =
      std::max<std::size_t>(1, std::max(displayWidth(alive_), displayWidth(dead_)));

  // Reserve one row for the header; clip the board to the visible viewport so a
  // grid larger than the terminal shows a top-left window rather than wrapping.
  const std::size_t visRows =
      std::min<std::size_t>(grid.rows(), termRows > 1 ? termRows - 1u : 1u);
  const std::size_t visCols = std::min<std::size_t>(grid.cols(), termCols / cellCols);
  const bool clipped = visRows < grid.rows() || visCols < grid.cols();

  const std::size_t glyph = std::max(alive_.size(), dead_.size()); // bytes per cell
  std::string out;
  out.reserve(visRows * (visCols * glyph + 4) + 64);
  out += kSyncBegin;
  out += kCursorHome;

  out += "generation ";
  out += std::to_string(generation);
  if (clipped) {
    out += " [";
    out += std::to_string(visCols);
    out += "x";
    out += std::to_string(visRows);
    out += " of ";
    out += std::to_string(grid.cols());
    out += "x";
    out += std::to_string(grid.rows());
    out += "]";
  }
  out += kClearToEol;
  out += '\n';

  for (std::size_t y = 0; y < visRows; ++y) {
    for (std::size_t x = 0; x < visCols; ++x) {
      out += (grid.at(x, y) ? alive_ : dead_);
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
}

} // namespace gol

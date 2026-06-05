/**
 * @file RleLoader.cpp
 * @brief Implementation of the RLE file parser.
 */

#include "gol/patterns/RleLoader.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>

namespace gol {

namespace {

/**
 * @brief Trim ASCII whitespace from both ends of a string.
 * @param s Input string.
 * @return Copy of @p s with leading/trailing whitespace removed.
 */
std::string trim(const std::string& s) {
  std::size_t b = 0;
  std::size_t e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

/**
 * @brief Parse the RLE header line, e.g. "x = 3, y = 3, rule = B36/S23".
 *
 * The rule field (if present) is ignored -- the variant rule lives in
 * LifeRules.hpp / kernel.cl, not in the data files.
 * @param line          Header line text.
 * @param[out] width    Receives the `x` dimension.
 * @param[out] height   Receives the `y` dimension.
 */
void parseHeader(const std::string& line, std::size_t& width, std::size_t& height) {
  std::size_t pos = 0;
  while (pos < line.size()) {
    std::size_t comma = line.find(',', pos);
    std::string token = line.substr(pos, comma == std::string::npos ? std::string::npos
                                                                     : comma - pos);
    std::size_t eq = token.find('=');
    if (eq != std::string::npos) {
      std::string key = trim(token.substr(0, eq));
      std::string val = trim(token.substr(eq + 1));
      if (key == "x") width = static_cast<std::size_t>(std::stoul(val));
      else if (key == "y") height = static_cast<std::size_t>(std::stoul(val));
    }
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
}

} // namespace

Pattern RleLoader::load(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("RleLoader: cannot open file: " + path);

  Pattern pat;
  std::string line;
  bool headerParsed = false;
  std::string body; // concatenated run-length-encoded payload (newlines dropped)

  while (std::getline(in, line)) {
    const std::string t = trim(line);
    if (t.empty()) continue;
    if (t[0] == '#') continue;                         // comment / metadata line
    if (!headerParsed && (t[0] == 'x' || t[0] == 'X')) {
      parseHeader(t, pat.width, pat.height);
      headerParsed = true;
      continue;
    }
    body += t;                                         // pattern payload
  }

  // Decode the payload. A token is <count><tag>, count defaulting to 1:
  //   b -> dead run (advance x)      o -> live run (emit cells)
  //   $ -> end of row (advance y)    ! -> end of pattern
  std::size_t x = 0;
  std::size_t y = 0;
  std::size_t count = 0;
  bool haveCount = false;
  for (const char ch : body) {
    if (std::isspace(static_cast<unsigned char>(ch))) continue;
    if (std::isdigit(static_cast<unsigned char>(ch))) {
      count = count * 10 + static_cast<std::size_t>(ch - '0');
      haveCount = true;
      continue;
    }
    const std::size_t n = haveCount ? count : 1;
    bool done = false;
    switch (ch) {
      case 'b':
        x += n;
        break;
      case 'o':
        for (std::size_t i = 0; i < n; ++i) pat.liveCells.emplace_back(x++, y);
        break;
      case '$':
        y += n;
        x = 0;
        break;
      case '!':
        done = true;
        break;
      default:
        break; // ignore anything unexpected
    }
    count = 0;
    haveCount = false;
    if (done) break;
  }

  // If the header omitted dimensions, infer the bounding box from the cells.
  if (pat.width == 0 || pat.height == 0) {
    std::size_t w = 0;
    std::size_t h = 0;
    for (const auto& [cx, cy] : pat.liveCells) {
      w = std::max(w, cx + 1);
      h = std::max(h, cy + 1);
    }
    if (pat.width == 0) pat.width = w;
    if (pat.height == 0) pat.height = h;
  }

  return pat;
}

} // namespace gol

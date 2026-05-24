#pragma once

#include <cstddef>
#include <vector>

class GameOfLife {
public:
  GameOfLife(std::size_t rows, std::size_t cols);

  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return cols_; }

  bool at(std::size_t r, std::size_t c) const;
  void set(std::size_t r, std::size_t c, bool alive);

  void step();

private:
  std::size_t rows_;
  std::size_t cols_;
  std::vector<unsigned char> current_;
  std::vector<unsigned char> next_;
};

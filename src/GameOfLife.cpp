#include "GameOfLife.h"

#include <stdexcept>

GameOfLife::GameOfLife(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), current_(rows * cols, 0),
      next_(rows * cols, 0) {}

bool GameOfLife::at(std::size_t r, std::size_t c) const {
  if (r >= rows_ || c >= cols_)
    throw std::out_of_range("GameOfLife::at out of range");
  return current_[r * cols_ + c] != 0;
}

void GameOfLife::set(std::size_t r, std::size_t c, bool alive) {
  if (r >= rows_ || c >= cols_)
    throw std::out_of_range("GameOfLife::set out of range");
  current_[r * cols_ + c] = alive ? 1 : 0;
}

void GameOfLife::step() {
  // TODO: implement sequential Conway's Game of Life update with the
  // assignment variation (birth on exactly 3 OR exactly 6 live neighbours,
  // survival on 2 or 3 live neighbours).
}

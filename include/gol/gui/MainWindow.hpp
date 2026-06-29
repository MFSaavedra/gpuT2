#pragma once

#include <QMainWindow>

#include "gol/Config.hpp"

class GolGlWidget;
class QLabel;

/**
 * @file MainWindow.hpp
 * @brief Top-level application window: the GL viewport plus a control panel.
 *
 * Hosts a GolGlWidget as the central viewport and a dockable panel of native Qt
 * widgets (play/pause, step, reseed, clear, speed, colour mode, wrap) wired to the
 * viewport's slots. A status bar shows the generation, FPS, and zoom.
 */
class MainWindow : public QMainWindow {
  Q_OBJECT

public:
  /**
   * @brief Build the window and its controls for a given run configuration.
   * @param cfg    Initial board/sim configuration.
   * @param parent Optional Qt parent.
   */
  explicit MainWindow(const gol::Config& cfg, QWidget* parent = nullptr);

private slots:
  /// @brief Refresh the status-bar labels from the viewport's per-frame stats.
  void onStats(unsigned long long generation, double fps, double zoom);

private:
  GolGlWidget* view_ = nullptr; ///< Central GL viewport.
  QLabel* statusLabel_ = nullptr; ///< Status-bar text (gen / fps / zoom).
};

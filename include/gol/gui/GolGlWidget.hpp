#pragma once

#include <memory>

#include <QElapsedTimer>
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLWidget>
#include <QPointF>
#include <QString>

#include "gol/Config.hpp"
#include "gol/Grid.hpp"
#include "gol/ISimEngine.hpp"

#ifdef GOL_HAVE_CUDA
#include "gol/engines/CudaEngine.hpp"
#include "gol/render/CudaGlInterop.hpp"
#endif

class QTimer;

/**
 * @file GolGlWidget.hpp
 * @brief OpenGL viewport that runs a Game of Life engine and displays it.
 *
 * Drives any ISimEngine (CPU or CUDA) and presents the board through one of two
 * paths, chosen at init:
 *   - Interop:    CUDA writes the current device buffer straight into a registered
 *                 PBO (device->device, no host round trip). Used when a CudaEngine
 *                 runs against a GL context on the same (NVIDIA) GPU.
 *   - HostUpload: the board is download()ed to host memory and uploaded to the
 *                 texture each frame. Used for the CPU engine, and as a fallback
 *                 when interop is unavailable (e.g. a GL context on an iGPU).
 *
 * The CPU path means the viewer builds and runs with no CUDA toolkit present.
 */
class GolGlWidget : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core {
  Q_OBJECT

public:
  /**
   * @brief Construct the viewport for a given run configuration.
   * @param cfg    Run configuration; engine kind, grid size, seed, edge mode.
   * @param parent Optional Qt parent.
   */
  explicit GolGlWidget(gol::Config cfg, QWidget* parent = nullptr);
  ~GolGlWidget() override;

  bool isPlaying() const { return playing_; }                   ///< @return Whether the sim is advancing.
  unsigned long long generation() const { return generation_; } ///< @return Current generation index.

public slots:
  void setPlaying(bool playing);   ///< Start/stop advancing the simulation.
  void togglePlaying();            ///< Flip play/pause.
  void stepOnce();                 ///< Advance exactly one generation (when paused).
  void reseed();                   ///< Re-randomise the board and reset the generation counter.
  void clearBoard();               ///< Set every cell dead and reset the generation counter.
  void setSpeed(int gensPerFrame); ///< Generations advanced per displayed frame.
  void setColorMode(int mode);     ///< 0 = binary, 1 = neighbour count.
  void setWrapEnabled(bool wrap);  ///< Toggle bounded/toroidal edges (rebuilds the engine).
  void fitView();                  ///< Reset zoom/pan to fit the whole board.
  void loadRle(const QString& path); ///< Load an RLE pattern, stamp it centred, and upload.

signals:
  /**
   * @brief Emitted each frame with display stats, for the HUD labels.
   * @param generation Current generation index.
   * @param fps        Smoothed frames per second.
   * @param zoom       Pixels per cell currently in use.
   */
  void statsChanged(unsigned long long generation, double fps, double zoom);

  /// @brief Emitted once after init with the backend description (engine + present mode).
  void backendInfo(const QString& description);

protected:
  void initializeGL() override;
  void resizeGL(int w, int h) override;
  void paintGL() override;
  void mousePressEvent(QMouseEvent* e) override;
  void mouseMoveEvent(QMouseEvent* e) override;
  void mouseReleaseEvent(QMouseEvent* e) override;
  void wheelEvent(QWheelEvent* e) override;
  void keyPressEvent(QKeyEvent* e) override;

private:
  /// @brief How the current board is moved into the display texture.
  enum class Present { Interop, HostUpload };

  void seedGrid();                          ///< Fill grid_ from the configured RLE, or a random fill.
  void rebuildEngine();                     ///< (Re)create the engine, preserving the current board.
  unsigned int compileProgram();            ///< Compile/link the display shaders; returns the GL program.
  unsigned int compileShader(unsigned int type, const char* path); ///< Compile one stage from a file.
  QPointF screenToBoard(QPointF posLogical) const; ///< Logical widget pixel -> board cell (float).
  void paintAt(QPointF posLogical, unsigned char value); ///< Set the cell under the cursor.
  QString backendDescription() const;       ///< "cpu / host-upload", "cuda / zero-copy interop", ...

  gol::Config cfg_;                          ///< Board/sim configuration.
  gol::Grid grid_;                           ///< Host board: seed, reseed/clear, and host-upload scratch.
  std::unique_ptr<gol::ISimEngine> engine_;  ///< The simulation (CPU or CUDA).
  Present present_ = Present::HostUpload;     ///< Active presentation path.

#ifdef GOL_HAVE_CUDA
  gol::CudaGlInterop interop_;               ///< CUDA<->GL bridge for the PBO (interop path only).
  gol::CudaEngine* cudaEngine_ = nullptr;    ///< Non-owning alias to engine_ when it is a CudaEngine.
#endif

  unsigned int pbo_ = 0;  ///< Pixel buffer object (interop path).
  unsigned int tex_ = 0;  ///< GL_R8UI texture sampled by the fragment shader.
  unsigned int vao_ = 0;  ///< Empty VAO required by core profile for the fullscreen triangle.
  unsigned int prog_ = 0; ///< Display shader program.

  int uViewSize_ = -1, uScale_ = -1, uCenter_ = -1, uBoard_ = -1, uColorMode_ = -1, uTex_ = -1;

  QTimer* timer_ = nullptr; ///< Drives repaint at ~60 Hz.
  bool playing_ = false;    ///< Whether step() runs each frame.
  int speed_ = 1;           ///< Generations per frame.
  int colorMode_ = 0;       ///< Current colour mode.
  unsigned long long generation_ = 0;    ///< Generation counter.
  unsigned long long reseedCounter_ = 0; ///< Varies the seed across reseeds.
  bool initFailed_ = false; ///< Set if the engine could not be built at all.

  // View transform (framebuffer pixels): cell = center + (pixel - viewSize/2) / scale.
  double scale_ = 1.0;  ///< Pixels per cell (zoom).
  QPointF center_;      ///< Board cell shown at the screen centre (pan).
  bool viewInitialised_ = false; ///< Set once the first frame sizes the view to fit.

  // Mouse interaction state.
  bool painting_ = false;        ///< Left-drag paint in progress.
  unsigned char paintValue_ = 1; ///< 1 = draw alive, 0 = erase (Shift).
  bool panning_ = false;         ///< Middle/right-drag pan in progress.
  QPointF lastDragPos_;          ///< Last cursor position during a pan (logical px).

  QElapsedTimer fpsClock_; ///< Inter-frame timer for the FPS estimate.
  double fps_ = 0.0;       ///< Smoothed FPS.
};

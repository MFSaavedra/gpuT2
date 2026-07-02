/**
 * @file GolGlWidget.cpp
 * @brief Implementation of the Game of Life OpenGL viewport (CPU + CUDA).
 */

#include "gol/gui/GolGlWidget.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <QCoreApplication>
#include <QDateTime>
#include <QImage>
#include <QKeyEvent>
#include <QMessageBox>
#include <QMouseEvent>
#include <QTimer>
#include <QWheelEvent>

#include "gol/engines/CpuEngine.hpp"
#include "gol/patterns/Pattern.hpp"
#include "gol/patterns/RleLoader.hpp"

#ifndef GOL_GUI_SHADER_DIR
#define GOL_GUI_SHADER_DIR "src/gui/shaders"
#endif

namespace {
std::string readFile(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("GolGlWidget: cannot open shader file: " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}
} // namespace

GolGlWidget::GolGlWidget(gol::Config cfg, QWidget* parent)
    : QOpenGLWidget(parent),
      cfg_(cfg),
      grid_(cfg.rows, cfg.cols),
      initialGrid_(cfg.rows, cfg.cols) {
  setFocusPolicy(Qt::StrongFocus); // receive key events
  seedGrid();
  initialGrid_ = grid_; // remember the seed as the reset point

  timer_ = new QTimer(this);
  connect(timer_, &QTimer::timeout, this, [this] { update(); });
  timer_->start(16); // ~60 Hz
}

GolGlWidget::~GolGlWidget() {
  // Release GL/CUDA resources with the context current.
  makeCurrent();
#ifdef GOL_HAVE_CUDA
  interop_.unregister();
#endif
  if (pbo_) glDeleteBuffers(1, &pbo_);
  if (tex_) glDeleteTextures(1, &tex_);
  if (ageTex_) glDeleteTextures(1, &ageTex_);
  if (vao_) glDeleteVertexArrays(1, &vao_);
  if (prog_) glDeleteProgram(prog_);
  engine_.reset();
  doneCurrent();
}

void GolGlWidget::seedGrid() {
  grid_.fill(0);
  if (cfg_.rlePath) {
    try {
      gol::Pattern p = gol::RleLoader::load(*cfg_.rlePath);
      const std::size_t ox = cfg_.cols > p.width ? (cfg_.cols - p.width) / 2 : 0;
      const std::size_t oy = cfg_.rows > p.height ? (cfg_.rows - p.height) / 2 : 0;
      p.applyTo(grid_, ox, oy);
      return;
    } catch (const std::exception& e) {
      std::fprintf(stderr, "[gol_gui] could not load RLE '%s' (%s); using a random seed\n",
                   cfg_.rlePath->c_str(), e.what());
    }
  }
  grid_.randomize(cfg_.seed);
}

void GolGlWidget::rebuildEngine() {
  // Preserve the current board across a rebuild (used when toggling wrap).
  if (engine_) engine_->download(grid_);

#ifdef GOL_HAVE_CUDA
  cudaEngine_ = nullptr;
  if (cfg_.engine == gol::EngineKind::Cuda) {
    auto e = std::make_unique<gol::CudaEngine>(cfg_.blockSize, cfg_.wrap, /*useShared=*/false);
    cudaEngine_ = e.get();
    engine_ = std::move(e);
  } else {
    engine_ = std::make_unique<gol::CpuEngine>(cfg_.threads, cfg_.wrap);
  }
#else
  // No CUDA in this build: always the CPU engine, whatever was requested.
  engine_ = std::make_unique<gol::CpuEngine>(cfg_.threads, cfg_.wrap);
#endif

  engine_->upload(grid_);
}

void GolGlWidget::initializeGL() {
  initializeOpenGLFunctions();

  // Report which GPU the GL context landed on (stderr, bypassing Qt's logger so it
  // always reaches the terminal).
  if (const char* v = reinterpret_cast<const char*>(glGetString(GL_VENDOR)))
    std::fprintf(stderr, "[gol_gui] GL_VENDOR   = %s\n", v);
  if (const char* r = reinterpret_cast<const char*>(glGetString(GL_RENDERER)))
    std::fprintf(stderr, "[gol_gui] GL_RENDERER = %s\n", r);
  std::fflush(stderr);

  prog_ = compileProgram();
  uViewSize_ = glGetUniformLocation(prog_, "uViewSize");
  uScale_ = glGetUniformLocation(prog_, "uScale");
  uCenter_ = glGetUniformLocation(prog_, "uCenter");
  uBoard_ = glGetUniformLocation(prog_, "uBoardSize");
  uColorMode_ = glGetUniformLocation(prog_, "uColorMode");
  uTex_ = glGetUniformLocation(prog_, "uBoard");
  uPalette_ = glGetUniformLocation(prog_, "uPalette");
  uAgeMax_ = glGetUniformLocation(prog_, "uAgeMax");
  uAge_ = glGetUniformLocation(prog_, "uAge");

  glGenVertexArrays(1, &vao_); // core profile needs a bound VAO even with no attributes

  const std::size_t bytes = grid_.size();

  // PBO used by the interop path (harmless if the host-upload path is chosen).
  glGenBuffers(1, &pbo_);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, static_cast<GLsizeiptr>(bytes), nullptr,
               GL_DYNAMIC_COPY);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

  // R8UI texture: one unsigned byte per cell, sampled with usampler2D/texelFetch.
  glGenTextures(1, &tex_);
  glBindTexture(GL_TEXTURE_2D, tex_);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R8UI, static_cast<GLsizei>(cfg_.cols),
               static_cast<GLsizei>(cfg_.rows), 0, GL_RED_INTEGER,
               GL_UNSIGNED_BYTE, nullptr);
  glBindTexture(GL_TEXTURE_2D, 0);

  // Age texture (R8UI): per-cell generations-survived, sampled by the age heatmap.
  // Same geometry/params as the board; filled host-side from age_ (see paintGL).
  glGenTextures(1, &ageTex_);
  glBindTexture(GL_TEXTURE_2D, ageTex_);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R8UI, static_cast<GLsizei>(cfg_.cols),
               static_cast<GLsizei>(cfg_.rows), 0, GL_RED_INTEGER,
               GL_UNSIGNED_BYTE, nullptr);
  glBindTexture(GL_TEXTURE_2D, 0);
  age_.assign(grid_.size(), 0);

  // Build the engine. The CUDA context (if any) is initialised here.
  try {
    rebuildEngine();
  } catch (const std::exception& e) {
    const QString msg = QStringLiteral("Could not initialise the simulation engine:\n  %1")
                            .arg(QString::fromLatin1(e.what()));
    std::fprintf(stderr, "\n[gol_gui] %s\n", msg.toLocal8Bit().constData());
    std::fflush(stderr);
    initFailed_ = true;
    QTimer::singleShot(0, this, [msg] {
      QMessageBox::critical(nullptr, QStringLiteral("Engine unavailable"), msg);
      QCoreApplication::quit();
    });
    return;
  }

  // Choose the presentation path. Default to host-upload (works everywhere); try
  // the zero-copy interop path only for a CUDA engine, and fall back gracefully if
  // the GL context is on a different GPU than CUDA (e.g. an Optimus iGPU context).
  present_ = Present::HostUpload;
#ifdef GOL_HAVE_CUDA
  if (cudaEngine_) {
    try {
      interop_.registerBuffer(pbo_, bytes);
      present_ = Present::Interop;
    } catch (const std::exception& e) {
      const char* r = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
      std::fprintf(stderr,
                   "[gol_gui] CUDA/GL interop unavailable on '%s' (%s);\n"
                   "          falling back to host-upload display. For the zero-copy\n"
                   "          path, launch via scripts/run_gui.sh (NVIDIA GL context).\n",
                   r ? r : "unknown", e.what());
      std::fflush(stderr);
      present_ = Present::HostUpload;
    }
  }
#endif

  emit backendInfo(backendDescription());
  fpsClock_.start();
}

QString GolGlWidget::backendDescription() const {
  const QString eng = engine_ ? QString::fromStdString(engine_->name()) : QStringLiteral("none");
  const QString mode = present_ == Present::Interop ? QStringLiteral("zero-copy interop")
                                                    : QStringLiteral("host-upload");
  return eng + QStringLiteral(" / ") + mode;
}

unsigned int GolGlWidget::compileShader(unsigned int type, const char* path) {
  const std::string src = readFile(std::string(GOL_GUI_SHADER_DIR) + "/" + path);
  const char* csrc = src.c_str();
  GLuint sh = glCreateShader(type);
  glShaderSource(sh, 1, &csrc, nullptr);
  glCompileShader(sh);
  GLint ok = 0;
  glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    GLint len = 0;
    glGetShaderiv(sh, GL_INFO_LOG_LENGTH, &len);
    std::string log(static_cast<std::size_t>(len), '\0');
    glGetShaderInfoLog(sh, len, nullptr, log.data());
    throw std::runtime_error(std::string("shader compile failed (") + path + "): " + log);
  }
  return sh;
}

unsigned int GolGlWidget::compileProgram() {
  GLuint vs = compileShader(GL_VERTEX_SHADER, "display.vert");
  GLuint fs = compileShader(GL_FRAGMENT_SHADER, "display.frag");
  GLuint prog = glCreateProgram();
  glAttachShader(prog, vs);
  glAttachShader(prog, fs);
  glLinkProgram(prog);
  GLint ok = 0;
  glGetProgramiv(prog, GL_LINK_STATUS, &ok);
  if (!ok) {
    GLint len = 0;
    glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
    std::string log(static_cast<std::size_t>(len), '\0');
    glGetProgramInfoLog(prog, len, nullptr, log.data());
    throw std::runtime_error("shader link failed: " + log);
  }
  glDeleteShader(vs);
  glDeleteShader(fs);
  return prog;
}

void GolGlWidget::resizeGL(int /*w*/, int /*h*/) {
  // View is sized from the framebuffer in paintGL (handles HiDPI uniformly).
}

void GolGlWidget::paintGL() {
  if (initFailed_) return; // a quit is already scheduled

  const double dpr = devicePixelRatioF();
  const int fbw = std::max(1, static_cast<int>(std::lround(width() * dpr)));
  const int fbh = std::max(1, static_cast<int>(std::lround(height() * dpr)));

  if (!viewInitialised_) {
    fitView();
    viewInitialised_ = true;
  }

  // Present the CURRENT board first, then advance at the end of the frame. This way a
  // cell painted between frames is displayed for the frame it was drawn (and only then
  // evolves) -- otherwise stepping would consume it before it was ever shown.

  // Move the current board into the display texture.
  glBindTexture(GL_TEXTURE_2D, tex_);
#ifdef GOL_HAVE_CUDA
  if (present_ == Present::Interop && cudaEngine_) {
    // Zero-copy: device->device into the PBO, then PBO -> texture.
    interop_.copyFromDevice(cudaEngine_->currentDeviceBuffer());
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, static_cast<GLsizei>(cfg_.cols),
                    static_cast<GLsizei>(cfg_.rows), GL_RED_INTEGER,
                    GL_UNSIGNED_BYTE, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  } else
#endif
  {
    // Host-upload: download to host memory, then upload to the texture.
    if (engine_) {
      engine_->download(grid_);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); // source from client memory, not a PBO
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, static_cast<GLsizei>(cfg_.cols),
                      static_cast<GLsizei>(cfg_.rows), GL_RED_INTEGER,
                      GL_UNSIGNED_BYTE, grid_.data());
    }
  }

  // Age heatmap: keep a host-side per-cell "generations survived" buffer. Update it
  // once per generation (ageGen_ tracks the last board it saw), so pausing does not
  // keep ageing cells and a re-presented frame is free. Only runs in age mode, so the
  // other modes never pay for it -- and the interop path stays zero-copy elsewhere.
  if (colorMode_ == 2) {
    if (!ageInit_ || ageGen_ != generation_) {
#ifdef GOL_HAVE_CUDA
      // The interop path leaves grid_ stale; pull the current board for the age update.
      // (The host-upload path already download()ed grid_ above.)
      if (present_ == Present::Interop && engine_) engine_->download(grid_);
#endif
      const unsigned char* b = grid_.data();
      for (std::size_t i = 0; i < age_.size(); ++i)
        age_[i] = b[i] ? static_cast<unsigned char>(std::min(age_[i] + 1, 255)) : 0;
      ageGen_ = generation_;
      ageInit_ = true;
      ageTexDirty_ = true;
    }
    if (ageTexDirty_) {
      glBindTexture(GL_TEXTURE_2D, ageTex_);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); // source from client memory, not a PBO
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, static_cast<GLsizei>(cfg_.cols),
                      static_cast<GLsizei>(cfg_.rows), GL_RED_INTEGER,
                      GL_UNSIGNED_BYTE, age_.data());
      ageTexDirty_ = false;
    }
  }

  glViewport(0, 0, fbw, fbh);
  glClearColor(0.05f, 0.05f, 0.06f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);

  glUseProgram(prog_);
  glUniform2f(uViewSize_, static_cast<float>(fbw), static_cast<float>(fbh));
  glUniform1f(uScale_, static_cast<float>(scale_));
  glUniform2f(uCenter_, static_cast<float>(center_.x()), static_cast<float>(center_.y()));
  glUniform2i(uBoard_, static_cast<GLint>(cfg_.cols), static_cast<GLint>(cfg_.rows));
  glUniform1i(uColorMode_, colorMode_);
  glUniform1i(uPalette_, palette_);
  glUniform1f(uAgeMax_, static_cast<float>(kAgeMax));

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, tex_);
  glUniform1i(uTex_, 0);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, ageTex_);
  glUniform1i(uAge_, 1);
  glActiveTexture(GL_TEXTURE0); // leave unit 0 active for tidiness

  glBindVertexArray(vao_);
  glDrawArrays(GL_TRIANGLES, 0, 3);
  glBindVertexArray(0);

  // FPS estimate (exponential smoothing) and HUD update.
  const double dt = fpsClock_.nsecsElapsed() / 1.0e6; // ms since last frame
  fpsClock_.restart();
  if (dt > 0.0) {
    const double inst = 1000.0 / dt;
    fps_ = fps_ > 0.0 ? 0.9 * fps_ + 0.1 * inst : inst;
  }
  emit statsChanged(generation_, fps_, scale_);

  // Advance for the NEXT frame. With a generation cap (--gens N), auto-pause on it.
  if (playing_ && engine_) {
    for (int i = 0; i < speed_; ++i) {
      engine_->step();
      ++generation_;
      if (cfg_.generations > 0 && generation_ >= cfg_.generations) {
        playing_ = false;
        break;
      }
    }
  }
}

QPointF GolGlWidget::screenToBoard(QPointF posLogical) const {
  const double dpr = devicePixelRatioF();
  const double fbw = width() * dpr;
  const double fbh = height() * dpr;
  const QPointF fb(posLogical.x() * dpr, posLogical.y() * dpr); // top-left origin
  return QPointF(center_.x() + (fb.x() - 0.5 * fbw) / scale_,
                 center_.y() + (fb.y() - 0.5 * fbh) / scale_);
}

void GolGlWidget::paintAt(QPointF posLogical, unsigned char value) {
  if (!engine_) return;
  const QPointF b = screenToBoard(posLogical);
  const int x = static_cast<int>(std::floor(b.x()));
  const int y = static_cast<int>(std::floor(b.y()));
  if (x < 0 || y < 0 || x >= static_cast<int>(cfg_.cols) || y >= static_cast<int>(cfg_.rows))
    return;
  engine_->pokeCell(static_cast<std::size_t>(x), static_cast<std::size_t>(y), value);
  if (colorMode_ == 2 && !age_.empty()) {
    // Reflect the edit in the age buffer immediately (a step does not run), so the
    // painted/erased cell shows up this frame. GL upload happens in paintGL.
    age_[static_cast<std::size_t>(y) * cfg_.cols + static_cast<std::size_t>(x)] =
        value ? 1 : 0;
    ageTexDirty_ = true;
  }
  update();
}

void GolGlWidget::fitView() {
  const double dpr = devicePixelRatioF();
  const double fbw = std::max(1.0, width() * dpr);
  const double fbh = std::max(1.0, height() * dpr);
  const double sx = fbw / static_cast<double>(cfg_.cols);
  const double sy = fbh / static_cast<double>(cfg_.rows);
  scale_ = std::min(sx, sy);
  center_ = QPointF(cfg_.cols / 2.0, cfg_.rows / 2.0);
  update();
}

// ---- transport / display slots -------------------------------------------------

void GolGlWidget::setPlaying(bool playing) { playing_ = playing; }
void GolGlWidget::togglePlaying() { playing_ = !playing_; }

void GolGlWidget::stepOnce() {
  if (!engine_) return;
  engine_->step();
  ++generation_;
  update();
}

void GolGlWidget::reseed() {
  grid_.randomize(cfg_.seed + (++reseedCounter_));
  initialGrid_ = grid_; // the new random board is the new reset point
  if (engine_) engine_->upload(grid_);
  generation_ = 0;
  resetAge();
  update();
}

void GolGlWidget::clearBoard() {
  grid_.fill(0);
  initialGrid_ = grid_; // an empty board is the new reset point
  if (engine_) engine_->upload(grid_);
  generation_ = 0;
  resetAge();
  update();
}

void GolGlWidget::resetToInitial() {
  if (!engine_) return;
  grid_ = initialGrid_; // restore the snapshot taken at generation 0
  engine_->upload(grid_);
  generation_ = 0;
  resetAge();
  update();
}

void GolGlWidget::loadRle(const QString& path) {
  if (path.isEmpty() || !engine_) return;
  try {
    gol::Pattern p = gol::RleLoader::load(path.toStdString());
    grid_.fill(0);
    const std::size_t ox = cfg_.cols > p.width ? (cfg_.cols - p.width) / 2 : 0;
    const std::size_t oy = cfg_.rows > p.height ? (cfg_.rows - p.height) / 2 : 0;
    p.applyTo(grid_, ox, oy);
    initialGrid_ = grid_; // the loaded pattern is the new reset point
    engine_->upload(grid_);
    generation_ = 0;
    resetAge();
    update();
  } catch (const std::exception& e) {
    QMessageBox::warning(this, QStringLiteral("Could not load pattern"),
                         QString::fromLatin1(e.what()));
  }
}

void GolGlWidget::setSpeed(int gensPerFrame) { speed_ = std::max(1, gensPerFrame); }

void GolGlWidget::setColorMode(int mode) {
  colorMode_ = mode;
  if (mode == 2) resetAge(); // start the heatmap fresh from the current board
  update();
}

void GolGlWidget::setPalette(int palette) {
  palette_ = palette;
  update();
}

// Host-only: zero the age buffer and mark it dirty. The GL upload happens on the next
// paintGL (which has the context current), so this is safe to call from any UI slot.
void GolGlWidget::resetAge() {
  std::fill(age_.begin(), age_.end(), static_cast<unsigned char>(0));
  ageInit_ = false;
  ageTexDirty_ = true;
}

void GolGlWidget::saveScreenshot() {
  const QImage img = grabFramebuffer(); // makes the context current internally
  const QString name =
      QStringLiteral("gol_shot_%1.png")
          .arg(QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_hhmmss_zzz")));
  if (img.save(name)) {
    std::fprintf(stderr, "[gol_gui] screenshot saved: %s\n", name.toLocal8Bit().constData());
    emit screenshotSaved(name);
  } else {
    std::fprintf(stderr, "[gol_gui] failed to save screenshot '%s'\n",
                 name.toLocal8Bit().constData());
  }
}

void GolGlWidget::setWrapEnabled(bool wrap) {
  if (wrap == cfg_.wrap) return;
  cfg_.wrap = wrap;
  makeCurrent();
  rebuildEngine(); // preserves the current board
  doneCurrent();
  update();
}

// ---- input ---------------------------------------------------------------------

void GolGlWidget::mousePressEvent(QMouseEvent* e) {
  if (e->button() == Qt::LeftButton) {
    painting_ = true;
    paintValue_ = (e->modifiers() & Qt::ShiftModifier) ? 0 : 1; // Shift erases
    paintAt(e->position(), paintValue_);
  } else if (e->button() == Qt::MiddleButton || e->button() == Qt::RightButton) {
    panning_ = true;
    lastDragPos_ = e->position();
  }
}

void GolGlWidget::mouseMoveEvent(QMouseEvent* e) {
  if (painting_) {
    paintAt(e->position(), paintValue_);
  } else if (panning_) {
    const double dpr = devicePixelRatioF();
    const QPointF d = e->position() - lastDragPos_;
    lastDragPos_ = e->position();
    // Drag moves the content with the cursor: shift the centre the opposite way.
    center_ -= QPointF(d.x() * dpr / scale_, d.y() * dpr / scale_);
    update();
  }
}

void GolGlWidget::mouseReleaseEvent(QMouseEvent* e) {
  if (e->button() == Qt::LeftButton) painting_ = false;
  if (e->button() == Qt::MiddleButton || e->button() == Qt::RightButton) panning_ = false;
}

void GolGlWidget::wheelEvent(QWheelEvent* e) {
  const double steps = e->angleDelta().y() / 120.0;
  if (steps == 0.0) return;
  const QPointF before = screenToBoard(e->position());
  scale_ *= std::pow(1.2, steps);
  scale_ = std::clamp(scale_, 0.02, 256.0);
  const QPointF after = screenToBoard(e->position());
  center_ += (before - after); // keep the cell under the cursor fixed
  update();
}

void GolGlWidget::keyPressEvent(QKeyEvent* e) {
  switch (e->key()) {
    case Qt::Key_Space: togglePlaying(); break;
    case Qt::Key_S:     stepOnce(); break;
    case Qt::Key_R:     reseed(); break;
    case Qt::Key_C:     clearBoard(); break;
    case Qt::Key_F:     fitView(); break;
    case Qt::Key_I:     resetToInitial(); break;
    case Qt::Key_P:     saveScreenshot(); break;
    default:            QOpenGLWidget::keyPressEvent(e); return;
  }
}

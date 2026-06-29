/**
 * @file MainWindow.cpp
 * @brief Implementation of the top-level window and its control panel.
 */

#include "gol/gui/MainWindow.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDockWidget>
#include <QFileDialog>
#include <QFormLayout>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QStatusBar>
#include <QVBoxLayout>
#include <QWidget>

#include "gol/gui/GolGlWidget.hpp"

MainWindow::MainWindow(const gol::Config& cfg, QWidget* parent)
    : QMainWindow(parent) {
  view_ = new GolGlWidget(cfg, this);
  setCentralWidget(view_);
  setWindowTitle("Game of Life — CUDA/OpenGL interop");
  resize(1100, 800);

  // ---- control panel (dockable) ----
  auto* panel = new QWidget;
  auto* form = new QFormLayout(panel);

  auto* playBtn = new QPushButton("Play / Pause");
  playBtn->setCheckable(true);
  auto* stepBtn = new QPushButton("Step");
  auto* reseedBtn = new QPushButton("Reseed");
  auto* clearBtn = new QPushButton("Clear");
  auto* openBtn = new QPushButton("Open RLE…");
  auto* fitBtn = new QPushButton("Fit view");

  auto* speed = new QSlider(Qt::Horizontal);
  speed->setRange(1, 50);
  speed->setValue(1);

  auto* colorMode = new QComboBox;
  colorMode->addItem("Binary");
  colorMode->addItem("Neighbour count");

  auto* wrap = new QCheckBox("Toroidal edges");
  wrap->setChecked(cfg.wrap);

  form->addRow(playBtn);
  form->addRow(stepBtn);
  form->addRow(reseedBtn);
  form->addRow(clearBtn);
  form->addRow(openBtn);
  form->addRow(fitBtn);
  form->addRow("Speed (gens/frame)", speed);
  form->addRow("Colour", colorMode);
  form->addRow(wrap);

  auto* dock = new QDockWidget("Controls", this);
  dock->setWidget(panel);
  dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
  addDockWidget(Qt::RightDockWidgetArea, dock);

  statusLabel_ = new QLabel;
  statusBar()->addPermanentWidget(statusLabel_);

  // ---- wiring ----
  connect(playBtn, &QPushButton::toggled, view_, &GolGlWidget::setPlaying);
  connect(stepBtn, &QPushButton::clicked, view_, &GolGlWidget::stepOnce);
  connect(reseedBtn, &QPushButton::clicked, view_, &GolGlWidget::reseed);
  connect(clearBtn, &QPushButton::clicked, view_, &GolGlWidget::clearBoard);
  connect(fitBtn, &QPushButton::clicked, view_, &GolGlWidget::fitView);
  connect(openBtn, &QPushButton::clicked, this, [this] {
    const QString f = QFileDialog::getOpenFileName(
        this, "Open RLE pattern", QString(), "RLE patterns (*.rle);;All files (*)");
    if (!f.isEmpty()) view_->loadRle(f);
  });
  connect(speed, &QSlider::valueChanged, view_, &GolGlWidget::setSpeed);
  connect(colorMode, &QComboBox::currentIndexChanged, view_, &GolGlWidget::setColorMode);
  connect(wrap, &QCheckBox::toggled, view_, &GolGlWidget::setWrapEnabled);
  connect(view_, &GolGlWidget::statsChanged, this, &MainWindow::onStats);
  connect(view_, &GolGlWidget::backendInfo, this, [this](const QString& d) {
    setWindowTitle("Game of Life — " + d);
  });
}

void MainWindow::onStats(unsigned long long generation, double fps, double zoom) {
  statusLabel_->setText(QString("gen %1   |   %2 FPS   |   %3 px/cell")
                            .arg(generation)
                            .arg(fps, 0, 'f', 0)
                            .arg(zoom, 0, 'f', 2));
}

#pragma once

#include <cstddef>
#include <string>

#include "gol/Grid.hpp"
#include "gol/ISimEngine.hpp"

namespace gol {

    class OpenCLEngine final : public ISimEngine {
    public:
        explicit OpenCLEngine(int blockSize = 256, bool wrap = false, bool useShared = false);
        ~OpenCLEngine() override;

        OpenCLEngine(const OpenCLEngine&) = delete;
        OpenCLEngine& operator=(const OpenCLEngine&) = delete;
        OpenCLEngine(OpenCLEngine&&) = delete;
        OpenCLEngine& operator=(OpenCLEngine&&) = delete;

        void upload(const Grid& initial) override;
        void step() override;
        void download(Grid& out) override;

        double lastKernelMillis() const override { return lastMs_; }
        std::string name() const override { return "opencl"; }

    private:
        int blockSize_;
        bool wrap_;
        bool useShared_;

        std::size_t rows_ = 0;
        std::size_t cols_ = 0;
        std::size_t bytes_ = 0;

        void* platform_ = nullptr;
        void* device_ = nullptr;
        void* context_ = nullptr;
        void* queue_ = nullptr;
        void* program_ = nullptr;
        void* kernelLocal_ = nullptr;
        void* kernelGlobal_ = nullptr;

        void* dCur_ = nullptr;
        void* dNxt_ = nullptr;

        double lastMs_ = 0.0;
    };

}
#pragma once
#include "latent_inpaint.h"
#include <memory>

namespace latent_inpaint {
class GraphPipeline {
public:
    GraphPipeline(const Options & options, const std::vector<int> & timesteps, const std::vector<float> & alphas);
    ~GraphPipeline();
    Tensor run(const Tensor & image, const Tensor & mask, const Tensor & posterior_image,
               const Tensor & posterior_masked, const Tensor & latent_noise, const Tensor & offset_noise);
#ifdef CRISPEMBED_LATENT_INPAINT_TESTING
    void set_observer(Observer callback, void * opaque);
#endif
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace latent_inpaint

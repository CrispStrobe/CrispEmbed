#pragma once
#include "latent_inpaint_ops.h"
#include <memory>
namespace lama {
class GraphModel {
public:
    GraphModel(const char * path, int threads);
    ~GraphModel();
    latent_inpaint::Tensor run(const latent_inpaint::Tensor & input);
#ifdef CRISPEMBED_LAMA_TESTING
    using Observer = void (*)(void *, const std::string &, const latent_inpaint::Tensor &);
    void set_observer(Observer callback, void * opaque);
#endif
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace lama

// Test-only observation ABI. Compile a separate lightonocr.cpp translation unit
// with CRISPEMBED_LIGHTONOCR3_TESTING; production archives have none of these symbols.
#pragma once
#include "lightonocr.h"
#include <cstddef>
#include <cstdint>
#ifdef CRISPEMBED_LIGHTONOCR3_TESTING
using lightonocr_test_callback = void (*)(const char *, const float *, size_t, void *);
void lightonocr_test_set_observer(lightonocr_context *, lightonocr_test_callback, void *);
void lightonocr_test_force_tokens(lightonocr_context *, const int32_t *, size_t);
bool lightonocr_test_is_profile3(const lightonocr_context *);
const int32_t * lightonocr_test_generated_ids(const lightonocr_context *, size_t *, bool argmax);
#endif

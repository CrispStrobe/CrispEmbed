#pragma once

// Shortest round-trip IEEE double conversion, independent of the host libc++
// floating to_chars availability (macOS 13.3 / iOS 16.3).
#ifdef __cplusplus
extern "C" {
#endif
int crispembed_ryu_d2s_buffered_n(double value, char * output);
#ifdef __cplusplus
}
#endif

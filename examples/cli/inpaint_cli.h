#pragma once

#include <cstdio>

// Kept separate from the embedding CLI so an inpainting request cannot fall
// through to an unrelated model, text input, or stdout format.
namespace inpaint_cli {

bool requested(int argc, char ** argv);
void print_help(std::FILE * out);
int run(int argc, char ** argv);

} // namespace inpaint_cli

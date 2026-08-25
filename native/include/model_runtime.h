#pragma once

// RKNN3 two-card runtime boundary.  This translation unit owns tokenizer,
// RKNN contexts, sessions, KV cache and generation.  Protocol handling lives
// in server.cc; main.cc never includes RKNN headers.
int model_runtime_run(int argc, char** argv);

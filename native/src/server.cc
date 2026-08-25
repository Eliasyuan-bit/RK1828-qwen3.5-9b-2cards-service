#include "server.h"

#include "model_runtime.h"

bool server_ctx::init(int argc, char** argv)
{
  argc_ = argc;
  argv_ = argv;
  return argc_ > 0 && argv_ != nullptr;
}

int server_ctx::run()
{
  // The runtime owns model lifetime. This layer has no RKNN API dependency and
  // can host a socket/HTTP transport in a later revision.
  return model_runtime_run(argc_, argv_);
}

void server_ctx::deinit()
{
  argc_ = 0;
  argv_ = nullptr;
}

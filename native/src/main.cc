#include "server.h"

int main(int argc, char** argv)
{
  server_ctx server;
  if (!server.init(argc, argv)) {
    return -1;
  }

  const int result = server.run();
  server.deinit();
  return result;
}

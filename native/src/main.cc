#include "service_runtime.h"

int main(int argc, char** argv)
{
  ServiceRuntime service;
  if (!service.init(argc, argv)) {
    return -1;
  }

  const int result = service.run();
  service.deinit();
  return result;
}

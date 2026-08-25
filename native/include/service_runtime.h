#pragma once

class ServiceRuntime {
 public:
  bool init(int argc, char** argv);
  int run();
  void deinit();

 private:
  int argc_ = 0;
  char** argv_ = nullptr;
};

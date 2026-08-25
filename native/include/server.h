#pragma once

// Process lifecycle for the JSONL service.  Keep main.cc limited to this
// public boundary so CLI/bootstrap code cannot reach model implementation.
class server_ctx {
 public:
  bool init(int argc, char** argv);
  int run();
  void deinit();

 private:
  int argc_ = 0;
  char** argv_ = nullptr;
};

#include <chrono>

#include "common.hpp"

class Timer {
 public:
  void start() { _start = std::chrono::high_resolution_clock::now(); }

  void end() { _end = std::chrono::high_resolution_clock::now(); }

  size_t getTimeNs() const { return std::chrono::duration_cast<std::chrono::nanoseconds>(_end - _start).count(); }

  size_t getTimeMs() const { return std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start).count(); }

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> _start;
  std::chrono::time_point<std::chrono::high_resolution_clock> _end;
};

#define TIME_EXECUTION(func)                             \
  {                                                      \
    Timer timer;                                         \
    timer.start();                                       \
    func;                                                \
    timer.end();                                         \
    show_info("{} took {}ms", #func, timer.getTimeMs()); \
  }

#define TIME_EXECUTION_R(func, result)                   \
  {                                                      \
    Timer timer;                                         \
    timer.start();                                       \
    result = func;                                       \
    timer.end();                                         \
    show_info("{} took {}ms", #func, timer.getTimeMs()); \
  }
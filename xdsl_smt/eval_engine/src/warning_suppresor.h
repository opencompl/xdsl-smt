#define SUPPRESS_WARNINGS_BEGIN                                     \
  _Pragma("clang diagnostic push")                                  \
  _Pragma("clang diagnostic ignored \"-Wimplicit-int-conversion\"") \
  _Pragma("clang diagnostic ignored \"-Wsign-conversion\"")         \
  _Pragma("clang diagnostic ignored \"-Wold-style-cast\"")          \
  _Pragma("clang diagnostic ignored \"-Wshadow\"")                  \
  _Pragma("clang diagnostic ignored \"-Wunused-parameter\"")

#define SUPPRESS_WARNINGS_END _Pragma("clang diagnostic pop")

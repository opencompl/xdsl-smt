#pragma once

#include <cstdio>
#include <functional>
#include <string_view>
#include <vector>

class Result {
public:
  Result() = default;

  Result(unsigned int s, unsigned int p, unsigned int e, bool solved)
      : sound(s), precise(p), exact(e) {
    unsolvedSound = !solved ? s : 0;
    unsolvedPrecise = !solved ? p : 0;
    unsolvedExact = !solved ? e : 0;
  }

  Result &operator+=(const Result &rhs) {
    sound += rhs.sound;
    precise += rhs.precise;
    exact += rhs.exact;
    unsolvedSound += rhs.unsolvedSound;
    unsolvedPrecise += rhs.unsolvedPrecise;
    unsolvedExact += rhs.unsolvedExact;

    return *this;
  }

  friend class Results;

private:
  unsigned int sound;
  unsigned int precise;
  unsigned int exact;
  unsigned int unsolvedSound;
  unsigned int unsolvedPrecise;
  unsigned int unsolvedExact;
};

class Results {
private:
  std::vector<Result> r;
  unsigned int cases = {};
  unsigned int unsolvedCases = {};

public:
  Results(unsigned int numFns) { r = std::vector<Result>(numFns); }

  void printMember(
      std::string_view name,
      const std::function<unsigned int(const Result &x)> &getter) const {
    printf("%s:\n[", name.data());
    for (auto x : r)
      printf("%d, ", getter(x));
    printf("]\n");
  }

  void print() const {
    printMember("sound", [](const Result &x) { return x.sound; });
    printMember("precise", [](const Result &x) { return x.precise; });
    printMember("exact", [](const Result &x) { return x.exact; });
    printMember("num_cases", [this](const Result &_) { return cases; });
    printMember("unsolved_sound",
                [](const Result &x) { return x.unsolvedSound; });
    printMember("unsolved_precise",
                [](const Result &x) { return x.unsolvedPrecise; });
    printMember("unsolved_exact",
                [](const Result &x) { return x.unsolvedExact; });
    printMember("unsolved_num_cases",
                [this](const Result &_) { return unsolvedCases; });
  }

  void incResult(const Result &newR, unsigned int i) { r[i] += newR; }

  void incCases(bool solved) {
    cases += 1;
    unsolvedCases += !solved ? 1 : 0;
  }
};

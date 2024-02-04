//
//  main.cpp
//  rootbit
//
//  Created by Etienne Pierre-doray on 2023-12-31.
//
//#include <numbers>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <numeric>
#include <vector>
#include <cmath>
#include <future>

#include <bitset>
#include <unordered_map>

using Bits = std::bitset<32>;

struct pair_hash {
  template <class T, class U>
  size_t operator() (const std::pair<T, U>& p) const {
    return ((size_t)std::hash<T>{}(p.first) << 32) | std::hash<U>{}(p.second);
  }
};

struct tuple_hash {
  template <class T, class U, class V>
  size_t operator() (const std::tuple<T, U, V>& p) const {
    return ((size_t)std::hash<T>{}(std::get<0>(p)) << 32) | std::hash<U>{}(std::get<1>(p)) | std::hash<V>{}(std::get<2>(p));
  }
};

Bits Keep(Bits value, size_t n) {
  return (value << (value.size() - n)) >> (value.size() - n);
}

Bits RightHalf(Bits value, size_t half_width) {
  return Keep(value, half_width);
}

Bits LeftHalf(Bits value, size_t half_width) {
  return value >> half_width;
}

size_t HammingDistance(Bits lhs, Bits rhs, size_t width, size_t symmetry) {
  if (width == 1) return lhs != rhs;
  size_t half_width = width / 2;
  size_t d1 =
    HammingDistance(RightHalf(lhs, half_width), RightHalf(rhs, half_width), half_width, symmetry) +
    HammingDistance(LeftHalf(lhs, half_width), LeftHalf(rhs, half_width), half_width, symmetry);
  if (width > symmetry) {
    return d1;
  }
  size_t d2 =
    HammingDistance(RightHalf(lhs, half_width), LeftHalf(rhs, half_width), half_width, symmetry) +
    HammingDistance(LeftHalf(lhs, half_width), RightHalf(rhs, half_width), half_width, symmetry);
  return std::min(d1, d2);
}

double Hamming1stMoment(Bits value, size_t width) {
  int moment = 0;
  int weight = 0;
  for (size_t i = 0; i < width; ++i) {
    if (value.test(i)) {
      moment += i;
      weight += 1;
    }
  }
  return double(moment) / double(weight);
}

double Hamming2ndMoment(Bits value, size_t width) {
  int moment = 0;
  int weight = 0;
  for (size_t i = 0; i < width; ++i) {
    if (value.test(i)) {
      moment += i*i;
      weight += 1;
    }
  }
  return double(moment) / double(weight);
}

std::vector<Bits> GenerateEquivalentClassesImpl(std::unordered_map<std::pair<size_t, size_t>, std::vector<Bits>, pair_hash>& table, size_t sum, size_t width, size_t symmetry) {
  if (width == 1) {
    return {Bits(sum)};
  }
  auto key = std::make_pair(sum, width);
  if (table.count(key)) {
    return table.at(key);
  }
  std::vector<Bits> result;
  size_t half_width = width / 2;
  for (size_t i = 0; i <= sum / 2; ++i) {
    if ((sum - i) > half_width) continue;
    if (i > half_width) continue;
    if (i == sum - i) {
      auto left_list = GenerateEquivalentClassesImpl(table, i, half_width, symmetry);
      for (size_t j = 0; j < left_list.size(); ++ j) {
        for (size_t k = 0; k <= j; ++k) {
          result.push_back(left_list[k] | (left_list[j] << half_width));
          if (width > symmetry && left_list[k] != left_list[j]) {
            result.push_back(left_list[j] | (left_list[k] << half_width));
          }
        }
      }
    } else {
      auto left_list = GenerateEquivalentClassesImpl(table, sum-i, half_width, symmetry);
      auto right_list = GenerateEquivalentClassesImpl(table, i, half_width, symmetry);
      for (auto left_entry : left_list) {
        for (auto right_entry : right_list) {
          result.push_back((left_entry << half_width) | right_entry);
          if (width > symmetry && left_entry != right_entry) {
            result.push_back((right_entry << half_width) | left_entry);
          }
        }
      }
    }
  }
  table[key] = result;
  return result;
}

std::vector<Bits> GenerateEquivalentClasses(size_t width, size_t symmetry) {
  std::unordered_map<std::pair<size_t, size_t>, std::vector<Bits>, pair_hash> table;
  std::vector<Bits> result;
  for (size_t i = 0; i <= width; ++i) {
    auto append = GenerateEquivalentClassesImpl(table, i, width, symmetry);
    result.insert(result.end(), append.begin(), append.end());
  }
  return result;
}

size_t GetEquivalentClassSize(Bits value, size_t width, size_t symmetry) {
  if (width == 1) return 1;
  size_t half_width = width / 2;
  size_t left_size = GetEquivalentClassSize(LeftHalf(value, half_width), half_width, symmetry);
  if (RightHalf(value, half_width) != LeftHalf(value, half_width)) {
    size_t right_size = GetEquivalentClassSize(RightHalf(value, half_width), half_width, symmetry);
    if (width <= symmetry) {
      return left_size * right_size * 2;
    }
    return left_size * right_size;
  }
  return left_size * left_size;
}

double logmul(double lhs, double rhs) {
  return lhs + rhs;
}

double fabs(double x) {
  if (x < 0) return -x;
  return x;
}

double logadd(double lhs, double rhs) {
  double d = fabs(lhs - rhs);
  double c = std::max(lhs, rhs);
  if (d > 30) {
    return c;
  }
  if (d > 9) {
    return c + exp(-d);
  }
  return c + log1p(exp(-d));
}

double logsub(double lhs, double rhs) {
  double c = std::min(lhs, rhs);
  return c + log(exp(lhs - c) - exp(rhs - c));
}

double logfactorial_impl(size_t n) {
  if (n == 0 || n == 1) return 0;
  return log(n) + logfactorial_impl(n-1);
}

double logfactorial(size_t n) {
  const static auto cache = []() {
    std::vector<double> c;
    for (size_t i = 0; i < 2048; ++i) {
      c.push_back(logfactorial_impl(i));
    }
    return c;
  }();
  return cache[n];
}

struct ProbValues {
  double p;
  double q;
  double p_1;
  double q_1;
};

using BaseTransitions = std::pair<std::array<double, 3>, std::array<double, 3>>;

BaseTransitions ComputeBaseConditionalProbability(ProbValues values) {
  std::array<double, 3> result_0 = {
    logmul(values.p_1, values.p_1),
    logmul(values.p_1, values.p),
    logmul(values.p, values.p)
  };
  std::array<double, 3> result_1 = {
    logmul(values.q, values.q),
    logmul(values.q_1, values.q),
    logmul(values.q_1, values.q_1)
  };
  return {std::move(result_0), std::move(result_1)};
}

double ComputeConditionalProbability(std::unordered_map<std::tuple<Bits, Bits, size_t>, double, tuple_hash>& memoized, size_t width, size_t symmetry, Bits repr, Bits parent_repr, ProbValues values) {
  if (width == 2) {
    if (parent_repr == 0b0) {
      if (repr == 0b00) {
        return logmul(values.p_1, values.p_1);
      } else if (repr == 0b11) {
        return logmul(values.p, values.p);
      }
      return logmul(values.p_1, values.p);
    } else {
      if (repr == 0b00) {
        return logmul(values.q, values.q);
      } else if (repr == 0b11) {
        return logmul(values.q_1, values.q_1);
      }
      return logmul(values.q_1, values.q);
    }
  }
  auto key = make_tuple(repr, parent_repr, width);
  if (memoized.count(key)) {
    return memoized.at(key);
  }
  size_t half_width = width / 2;
  auto left_parent = LeftHalf(parent_repr, width / 4);
  auto rigth_parent = RightHalf(parent_repr, width / 4);
  
  auto left_repr = LeftHalf(repr, half_width);
  auto rigth_repr = RightHalf(repr, half_width);
  
  auto probability_left = ComputeConditionalProbability(memoized, half_width, symmetry, left_repr, left_parent, values);
  auto probability_right = ComputeConditionalProbability(memoized, half_width, symmetry, rigth_repr, rigth_parent, values);
  double probability = logmul(probability_left, probability_right);
  if (left_parent != rigth_parent && width <= symmetry) {
    auto cross_probability_left = ComputeConditionalProbability(memoized, half_width, symmetry, left_repr, rigth_parent, values);
    auto cross_probability_right = ComputeConditionalProbability(memoized, half_width, symmetry, rigth_repr, left_parent, values);
    double cross_probability = logmul(cross_probability_left, cross_probability_right);
    probability = logmul(logadd(probability, cross_probability), log(0.5));
  }
  memoized[key] = probability;
  return probability;
}

std::unordered_map<Bits, double> ComputeProbability(const std::unordered_map<Bits, double>& parents, std::unordered_map<std::tuple<Bits, Bits, size_t>, double, tuple_hash>& memoized, size_t width, size_t symmetry, ProbValues values) {
  auto classes = GenerateEquivalentClasses(width, symmetry);
  
  std::unordered_map<Bits, double> result;
  for (auto repr : classes) {
    double total_probability = -std::numeric_limits<double>::infinity();
    for (auto [parent_repr, parent_probability] : parents) {
      if (parent_probability == -std::numeric_limits<double>::infinity()) continue;
      
      double conditional = ComputeConditionalProbability(memoized, width, symmetry, repr, parent_repr, values);
      double probability = logmul(parent_probability, conditional);
      if (total_probability == -std::numeric_limits<double>::infinity()) {
        total_probability = probability;
      } else {
        total_probability = logadd(total_probability, probability);
      }
    }
    size_t coefficient = GetEquivalentClassSize(repr, width, symmetry);
    total_probability = logmul(total_probability, log(double(coefficient)));
    result.emplace(repr, total_probability);
  }
  return result;
}

template <class T>
T mean(const std::vector<T> &vec) {
  const size_t sz = vec.size();
  if (sz <= 1) {
    return 0.0;
  }
  T mean = 0.0;
  for (size_t i = 0; i < vec.size(); ++i) {
    mean += i * exp(vec[i]);
  }
  return mean;
}

template <class T>
T variance(const std::vector<T> &vec) {
  const size_t sz = vec.size();
  if (sz <= 1) {
    return 0.0;
  }

  // Calculate the mean
  const T m = mean(vec);
  
  T variance = 0.0;
  for (size_t i = 0; i < vec.size(); ++i) {
    variance += ((i - m) * (i - m)) * exp(vec[i]);
  }
  return variance;
}

double GreedyMaxCoupling(const std::unordered_map<Bits, double>& probs_a, const std::unordered_map<Bits, double>& probs_b) {
  double sum_overlap = -std::numeric_limits<double>::infinity();
  for (const auto& k_v : probs_a) {
    double prob_a = k_v.second;
    if (!probs_b.count(k_v.first)) {
      continue;
    }
    double prob_b = probs_b.at(k_v.first);
    if (prob_a == -std::numeric_limits<double>::infinity() || prob_b == -std::numeric_limits<double>::infinity()) continue;
    double overlap = std::min(prob_a, prob_b);
    if (sum_overlap == -std::numeric_limits<double>::infinity()) {
      sum_overlap = overlap;
    } else if (overlap != -std::numeric_limits<double>::infinity()) {
      sum_overlap = logadd(sum_overlap, overlap);
    }
  }
  return exp(sum_overlap);
}

int main(int argc, const char * argv[]) {
  double p = 0.1;//0.5 - sqrt(1.0/8)-1.0/64;
  double q = 0.15;
  ProbValues values {log(p), log(q), log(1-p), log(1-q)};
  
  std::unordered_map<Bits, double> probs_a = {{0, 0}};
  std::unordered_map<Bits, double> probs_b = {{1, 0}};
  for (int i = 1; i <= 5; ++i) {
    std::unordered_map<std::tuple<Bits, Bits, size_t>, double, tuple_hash> memoized;
    int width = pow(2, i);
    std::cout << width << std::endl;
    probs_a = ComputeProbability(probs_a, memoized, width, width, values);
    probs_b = ComputeProbability(probs_b, memoized, width, width, values);
    
    std::vector<std::unordered_map<Bits, std::pair<double, double>>> bucketed_probs(width+1);
    for (const auto& k_v : probs_a) {
      bucketed_probs[k_v.first.count()].emplace(k_v.first, std::make_pair(k_v.second, probs_b.at(k_v.first)));
    }
    for (size_t i = 0; i < bucketed_probs.size(); ++i) {
      for (const auto& k_v : bucketed_probs[i]) {
        double min_prob = std::min(exp(k_v.second.first), exp(k_v.second.second));
        double prob_a = exp(k_v.second.first) - min_prob;
        double prob_b = exp(k_v.second.second) - min_prob;
        double moment_1 = Hamming1stMoment(k_v.first, width);
        double moment_2 = Hamming2ndMoment(k_v.first, width) - moment_1 * moment_1;
        std::cout << i << ", " << GetEquivalentClassSize(k_v.first, width, width) << ", " << k_v.first << ", " << Keep(~k_v.first, width) << ", " << HammingDistance(k_v.first, Keep(~k_v.first, width), width, width) << ", " << moment_1 << ", " << moment_2 << ", " << exp(k_v.second.first) << ", " << exp(k_v.second.second) << std::endl;
      }
    }
    for (size_t i = 0; i < bucketed_probs.size(); ++i) {
      std::cout << i << " " << bucketed_probs[i].size() << std::endl;
    }
  }
  return 0;
}

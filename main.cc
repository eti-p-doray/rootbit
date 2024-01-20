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

using Bits = std::bitset<8>;

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

Bits RightHalf(Bits value, size_t half_width) {
  return value & Bits((1<<half_width) - 1);
}

Bits LeftHalf(Bits value, size_t half_width) {
  return value >> half_width;
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
  /*if (memoized.count(key)) {
    return memoized.at(key);
  }*/
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
      std::cout << repr << " " << parent_repr << " " << exp(conditional) << " " << exp(probability) << std::endl;
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

double ComputeOverlapProbability(size_t width, ProbValues values) {
  auto base_cases = ComputeBaseConditionalProbability(values);
  double sum_overlap = -std::numeric_limits<double>::infinity();
  for (size_t i = 0; i <= width; ++i) {
    for (size_t j = 0; j <= width-i; ++j) {
      size_t k = width - i - j;
      double prob_0 = base_cases.first[0] * i + (base_cases.first[1] + log(2.0)) * j + base_cases.first[2] * k;
      double prob_1 = base_cases.second[0] * i + (base_cases.second[1] + log(2.0)) * j + base_cases.second[2] * k;
      double weight = logfactorial(width) - logfactorial(i) - logfactorial(j) - logfactorial(k);
      double overlap = std::min(prob_0, prob_1) + weight;
      if (sum_overlap == -std::numeric_limits<double>::infinity()) {
        sum_overlap = overlap;
      } else if (overlap != -std::numeric_limits<double>::infinity()) {
        sum_overlap = logadd(sum_overlap, overlap);
      }
    }
  }
  return exp(sum_overlap);
}

using HammingConditionalPart = double[4096 * 4096];

double ComputeConditionalHammingProbabilityPart1(int v, int parent_weight, const BaseTransitions& base_cases) {
  // (i,j,k) describe all the ways parent 1s can produce u, with i:1->00, j:1->11 and k:1->01.
  // i + j + k = parent_weight
  // k + 2*j = v
  // i,j,k >= 0
  double running_probability = -std::numeric_limits<double>::infinity();
  for (int i = std::max(parent_weight - v, 0); i <= parent_weight - v / 2.0; ++i) {
    int j = v - parent_weight + i;
    int k = v - 2 * j;
    //std::cout << "    " << i << " " << j << " " << k << std::endl;
    //assert(i+j+k == parent_weight);
    //assert(k+2*j == v);
    //assert(i >= 0 && j >= 0 && k >= 0);
    double prob_a = base_cases.second[0] * i + base_cases.second[2] * j + (base_cases.second[1] + log(2.0)) * k;
    double weight_a = logfactorial(parent_weight) - logfactorial(i) - logfactorial(j) - logfactorial(k);
    double probability = prob_a + weight_a;
    if (running_probability == -std::numeric_limits<double>::infinity()) {
      running_probability = probability;
    } else if (probability != -std::numeric_limits<double>::infinity()) {
      running_probability = logadd(running_probability, probability);
    }
  }
  return running_probability;
}

double ComputeConditionalHammingProbabilityPart0(int u, int parent_weight, const BaseTransitions& base_cases) {
  // (f,g,h) describe all the ways parent 0s can produce v, with f:0->00, g:0->11 and h:0->01.
  // f + g + h = width - parent_weight
  // h + 2*g = u
  // f,g,h >= 0
  double running_probability = -std::numeric_limits<double>::infinity();
  for (int f = std::max(parent_weight - u, 0); f <= parent_weight - u / 2.0; ++f) {
    int g = u - parent_weight + f;
    int h = u - 2 * g;
    //std::cout << "      " << f << " " << g << " " << h << std::endl;
    //assert(f+g+h == width - parent_weight);
    //assert(h+2*g == u);
    //assert(f >= 0 && g >= 0 && h >= 0);
    double prob_b = base_cases.first[0] * f + base_cases.first[2] * g + (base_cases.first[1] + log(2.0)) * h;
    double weight_b = logfactorial(parent_weight) - logfactorial(f) - logfactorial(g) - logfactorial(h);
    double probability = prob_b + weight_b;
    if (running_probability == -std::numeric_limits<double>::infinity()) {
      running_probability = probability;
    } else if (probability != -std::numeric_limits<double>::infinity()) {
      running_probability = logadd(running_probability, probability);
    }
  }
  return running_probability;
}

double ComputeConditionalHammingProbability(double table0[], double table1[], int width, int weight, int parent_weight) {
  double total_probability = -std::numeric_limits<double>::infinity();
  // The parent string looks like 00..011.1 with `parent_weight` 1s and `width-parent_weight` 0s.

  // (u,v) splits the hamming weight into weight coming from parent 0s and 1s respectively.
  // u + v = weight
  for (int u = std::max(weight-2*parent_weight, 0); u <= std::min(weight, 2*(width - parent_weight)); ++u) {
    int v = weight - u;
    
    double probability = table1[v] + table0[u];
    if (total_probability == -std::numeric_limits<double>::infinity()) {
      total_probability = probability;
    } else if (probability != -std::numeric_limits<double>::infinity()) {
      total_probability = logadd(total_probability, probability);
    }
  }
  return total_probability;
}


std::vector<double> ComputeHammingProbabilities(const std::vector<double>& parents, size_t width, ProbValues values) {
  auto transitions = ComputeBaseConditionalProbability(values);

  std::vector<double> results(width+1, -std::numeric_limits<double>::infinity());
  int half_width = int(width / 2);
  for (int j = 0; j < parents.size(); ++j) {
    if (parents[j] == -std::numeric_limits<double>::infinity()) continue;

    std::vector<double> table0(width+1, 0.0);
    std::vector<double> table1(width+1, 0.0);
    for (int k = 0; k <= width; ++k) {
      table1[k] = ComputeConditionalHammingProbabilityPart1(k, j, transitions);
      table0[k] = ComputeConditionalHammingProbabilityPart0(k, half_width-j, transitions);
    }

    for (int i = 0; i <= width; ++i) {
      if (parents[j] < results[i] - 30.0) continue;
      double probability = ComputeConditionalHammingProbability(table0.data(), table1.data(), half_width, i, j) + parents[j];
      if (results[i] == -std::numeric_limits<double>::infinity()) {
        results[i] = probability;
      } else if (probability != -std::numeric_limits<double>::infinity()) {
        results[i] = logadd(results[i], probability);
      }
    }
  }
  return results;
}

double GreedyMaxCoupling(const std::vector<double>& probs_a, const std::vector<double>& probs_b) {
  double sum_overlap = -std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < probs_a.size(); ++i) {
    double prob_a = probs_a[i];
    double prob_b = probs_b[i];
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

int main(int argc, const char * argv[]) {
  double p = 0.125;//0.5 - sqrt(1.0/8)-1.0/64;
  double q = p;
  ProbValues values {log(p), log(q), log(1-p), log(1-q)};
  
  std::vector<double> probs_a = {-std::numeric_limits<double>::infinity(), log(1)};
  std::vector<double> probs_b = {log(1), -std::numeric_limits<double>::infinity()};
  for (int i = 1; i <= 16; ++i) {
    int width = pow(2, i);
    //std::cout << width << std::endl;
    probs_a = ComputeHammingProbabilities(probs_a, width, values);
    probs_b = ComputeHammingProbabilities(probs_b, width, values);
    auto mean_a = mean(probs_a);
    auto variance_a = variance(probs_a);
    //auto z_a = (mean_a - width / 2) /
    auto mean_b = mean(probs_b);
    auto variance_b = variance(probs_b);
    //std::cout << variance_a << " " << variance_a / width << std::endl;
    //std::cout << variance_b << " " << variance_b / width << std::endl;
    //std::cout << mean_b << " " << variance_b << " " << mean_b - width/2  << std::endl;
    //std::cout << width << ", " << mean_a - width * p / (p + q)  << ", " << mean_b - width * p / (p + q) << std::endl;
    /*for (size_t j = 0; j < probs_a.size(); ++j) {
      std::cout << " " << j << ", " << exp(probs_a[j]) << ", " << exp(probs_b[j]) << std::endl;
    } */   
    double tv = GreedyMaxCoupling(probs_a, probs_b);
    std::cout << width << " " << tv << " " << mean_a << " " << mean_b << " " << variance_a << " " << variance_b << std::endl;
  }
  return 0;
}

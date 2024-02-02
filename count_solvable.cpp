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
  if (d > 40.0) {
    return c;
  }
  if (d > 10.0) {
    return c + exp(-d);
  }
  return c + log1p(exp(-d));
}

double logsub(double lhs, double rhs) {
  double c = std::min(lhs, rhs);
  return c + log(exp(lhs - c) - exp(rhs - c));
}

double logfactorial(size_t n) {
  const static auto cache = []() {
    std::vector<double> c;
    c.push_back(0);
    for (size_t i = 1; i < 65536; ++i) {
      c.push_back(c.back() + log(i));
    }
    return c;
  }();
  return cache[n];
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

double ComputeConditionalHammingProbabilityPart1(int v, int parent_weight, const ProbValues& values) {
  int n = 2 * parent_weight;
  return values.q_1 * v + (values.q) * (n - v) + logfactorial(n) - logfactorial(v) - logfactorial(n - v);
}

double ComputeConditionalHammingProbabilityPart0(int u, int parent_weight, const ProbValues& values) {
  int n = 2 * parent_weight;
  return values.p * u + (values.p_1) * (n - u) + logfactorial(n) - logfactorial(u) - logfactorial(n - u);
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


std::vector<double> ComputeHammingProbabilitiesImpl(const std::vector<double>& parents, int lo, int hi, size_t width, ProbValues values) {
  auto transitions = ComputeBaseConditionalProbability(values);

  std::vector<double> results(width+1, -std::numeric_limits<double>::infinity());
  int half_width = int(width / 2);
  for (int j = lo; j < hi; ++j) {
    if (parents[j] == -std::numeric_limits<double>::infinity()) continue;

    //std::cout << "j: " << j << " : " << width << std::endl;
    std::vector<double> table0(width+1, 0.0);
    std::vector<double> table1(width+1, 0.0);
    for (int k = 0; k <= width; ++k) {
      table1[k] = ComputeConditionalHammingProbabilityPart1(k, j, values);
      table0[k] = ComputeConditionalHammingProbabilityPart0(k, half_width-j, values);
      
      //std::cout << exp(table0[k]) << ", " << exp(table1[k]) << std::endl;
    }

    std::vector<double> conv;
    for (int i = 0; i <= width; ++i) {
      //if (parents[j] < results[i] - 40.0) continue;
      double probability = ComputeConditionalHammingProbability(table0.data(), table1.data(), half_width, i, j);
      conv.push_back(probability);
      //std::cout << exp(probability) << std::endl;
      probability += parents[j];
      if (results[i] == -std::numeric_limits<double>::infinity()) {
        results[i] = probability;
      } else if (probability != -std::numeric_limits<double>::infinity()) {
        results[i] = logadd(results[i], probability);
      }
    }
    double m = mean(conv);
    double v = variance(conv);
    
    //std::cout << j << ", " << m << ", " << v << std::endl;
    //std::cout << 2.0 * ((half_width - j) * exp(values.p) + j * exp(values.q_1)) << " " << 2.0 * (j * exp(values.q) * exp(values.q_1) + (half_width - j) * exp(values.p) * exp(values.p_1)) << std::endl;
  }
  return results;
}

std::vector<double> ComputeHammingProbabilities(const std::vector<double>& parents, size_t width, ProbValues values) {
  return ComputeHammingProbabilitiesImpl(parents, 0, parents.size(), width, values);
  /*std::vector<std::future<std::vector<double>>> thread_pool;
  size_t num_threads = 16;
  size_t batch_size = std::max<size_t>(parents.size() / num_threads, 1U);
  for (int lo = 0; lo < parents.size(); lo += batch_size) {
    int hi = std::min(lo + batch_size, parents.size());
    //std::cout << lo << " " << hi << std::endl;
    thread_pool.emplace_back(std::async(std::launch::async, ComputeHammingProbabilitiesImpl, parents, lo, hi, width, values));
  }
  std::vector<double> results(width+1, -std::numeric_limits<double>::infinity());
  for (auto& f: thread_pool) {
    auto result = f.get();
    for (size_t i = 0; i < results.size(); ++i) {
      if (results[i] == -std::numeric_limits<double>::infinity()) {
        results[i] = result[i];
      } else if (result[i] != -std::numeric_limits<double>::infinity()) {
        results[i] = logadd(results[i], result[i]);
      }
    }
  }
  return results;*/
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

int main(int argc, const char * argv[]) {
  double p = ( 1.0/2 - sqrt(1.0/8) );//0.12;//0.5 - sqrt(1.0/8)-1.0/64;
  double q = ( 1.0/2 - sqrt(1.0/8) );
  ProbValues values {log(p), log(q), log(1-p), log(1-q)};
  
  std::vector<double> probs_a = {-std::numeric_limits<double>::infinity(), log(1)};
  std::vector<double> probs_b = {log(1), -std::numeric_limits<double>::infinity()};
  for (int i = 1; i <= 7; ++i) {
    int width = pow(2, i);
    probs_a = ComputeHammingProbabilities(probs_a, width, values);
    probs_b = ComputeHammingProbabilities(probs_b, width, values);
    auto mean_a = mean(probs_a);
    auto variance_a = variance(probs_a);
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

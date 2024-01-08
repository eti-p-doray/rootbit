//
//  main.cpp
//  rootbit
//
//  Created by Etienne Pierre-doray on 2023-12-31.
//
#include <numbers>
#include <iostream>
#include <cassert>
#include <numeric>

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

double logadd(double lhs, double rhs) {
  double c = std::min(lhs, rhs);
  return c + log(exp(lhs - c) + exp(rhs - c));
  //return std::max(lhs, rhs);
}

double logsub(double lhs, double rhs) {
  double c = std::min(lhs, rhs);
  return c + log(exp(lhs - c) - exp(rhs - c));
}

double logfactorial_impl(size_t n) {
  if (n == 0 || n == 1) return 0;
  return log(n) + logfactorial_impl(n-1);
}

/*double logfactorial_impl(size_t n) {
  if (n == 0 || n == 1) return 0;
  //return log(tgamma(n+1));
  double x = n;
  return x * log(x) - x + (log(x * (1.0 + 4.0*x*(1.0+2.0*x)))) / 6.0 + log(std::numbers::pi) / 2.0;
}*/

double logfactorial(size_t n) {
  const static auto cache = []() {
    std::vector<double> c;
    for (size_t i = 0; i < 1024; ++i) {
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

std::pair<std::vector<double>, std::vector<double>> ComputeBaseConditionalProbability(ProbValues values) {
  std::vector<double> result_0 = {
    logmul(values.p_1, values.p_1),
    logmul(values.p_1, values.p),
    logmul(values.p, values.p)
  };
  std::vector<double> result_1 = {
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

double ComputeConditionalHammingProbability(int width, int weight, int parent_weight, std::pair<std::vector<double>, std::vector<double>> base_cases) {
  double total_probability = -std::numeric_limits<double>::infinity();
  // u + v = weight
  for (int u = std::max(weight-2*parent_weight, 0); u <= std::min(weight, 2*(width - parent_weight)); ++u) {
    int v = weight - u;
    
    // i + j + k = parent_weight
    // k + 2*j = v
    // i,j,k >= 0
    // f + g + h = width - parent_weight
    // h + 2*g = u
    // f,g,h >= 0
    for (int i = std::max(parent_weight - v, 0); i <= parent_weight - v / 2.0; ++i) {
      int j = v - parent_weight + i;
      int k = v - 2 * j;
      assert(i+j+k == parent_weight);
      assert(k+2*j == v);
      assert(i >= 0 && j >= 0 && k >= 0);
      double prob_a = base_cases.second[0] * i + base_cases.second[2] * j + (base_cases.second[1] + log(2.0)) * k;
      double weight_a = logfactorial(parent_weight) - logfactorial(i) - logfactorial(j) - logfactorial(k);
      
      for (int f = std::max(width - parent_weight - u, 0); f <= width - parent_weight - u / 2.0; ++f) {
        int g = u - width + parent_weight + f;
        int h = u - 2 * g;
        assert(f+g+h == width - parent_weight);
        assert(h+2*g == u);
        assert(f >= 0 && g >= 0 && h >= 0);
        double prob_b = base_cases.first[0] * f + base_cases.first[2] * g + (base_cases.first[1] + log(2.0)) * h;
        double weight_b = logfactorial(width - parent_weight) - logfactorial(f) - logfactorial(g) - logfactorial(h);
        double probability = prob_a + prob_b + weight_a + weight_b;
        if (total_probability == -std::numeric_limits<double>::infinity()) {
          total_probability = probability;
        } else if (probability != -std::numeric_limits<double>::infinity()) {
          total_probability = logadd(total_probability, probability);
        }
      }
    }
  }
  return total_probability;
}

std::vector<double> ComputeHammingProbability(const std::vector<double>& parents, size_t width, ProbValues values) {
  auto base_cases = ComputeBaseConditionalProbability(values);
  std::vector<double> results;
  int half_width = int(width / 2);
  for (int i = 0; i <= width; ++i) {
    double total_probability = -std::numeric_limits<double>::infinity();
    for (int j = 0; j < parents.size(); ++j) {
      if (parents[j] == -std::numeric_limits<double>::infinity()) continue;
      double probability = ComputeConditionalHammingProbability(half_width, i, j, base_cases) + parents[j];
      //std::cout << " " << i << " " << j << " " << probability << std::endl;
      if (total_probability == -std::numeric_limits<double>::infinity()) {
        total_probability = probability;
      } else {
        total_probability = logadd(total_probability, probability);
      }
    }
    results.push_back(total_probability);
  }
  return results;
}

double ComputeTotalVariation(const std::vector<double>& probs_a, const std::vector<double>& probs_b) {
  double total_variation = -std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < probs_a.size(); ++i) {
    for (size_t j = 0; j < probs_b.size(); ++j) {
      if (i == j) continue;
      double probability = probs_a[i] + probs_b[j];
      if (total_variation == -std::numeric_limits<double>::infinity()) {
        total_variation = probability;
      } else {
        total_variation = logadd(total_variation, probability);
      }
    }
  }
  return total_variation;
}

double GreedyMaxCoupling(const std::unordered_map<Bits, double> probs_a, const std::unordered_map<Bits, double> probs_b) {
  double sum_overlap = -std::numeric_limits<double>::infinity();
  for (auto [repr, prob_a] : probs_a) {
    if (!probs_b.count(repr)) continue;
    double prob_b = probs_b.at(repr);
    if (prob_a == -std::numeric_limits<double>::infinity() || prob_b == -std::numeric_limits<double>::infinity()) continue;
    double overlap = std::min(prob_a, prob_b);
    if (sum_overlap == -std::numeric_limits<double>::infinity()) {
      sum_overlap = overlap;
    } else {
      sum_overlap = logadd(sum_overlap, overlap);
    }
  }
  return exp(sum_overlap);
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
    } else {
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
  double p = 0.20;
  double q = 0.20;
  ProbValues values {log(p), log(q), log(1-p), log(1-q)};
  
  std::vector<double> probs_a = {-std::numeric_limits<double>::infinity(), log(1)};
  std::vector<double> probs_b = {log(1), -std::numeric_limits<double>::infinity()};
  for (int i = 1; i <= 12; ++i) {
    int width = pow(2, i);
    //std::cout << width << std::endl;
    probs_a = ComputeHammingProbability(probs_a, width, values);
    probs_b = ComputeHammingProbability(probs_b, width, values);
    auto mean_a = mean(probs_a);
    auto variance_a = variance(probs_a);
    //auto z_a = (mean_a - width / 2) /
    auto mean_b = mean(probs_b);
    auto variance_b = variance(probs_b);
    std::cout << variance_a << " " << variance_a / width << std::endl;
    std::cout << variance_b << " " << variance_b / width << std::endl;
    //std::cout << mean_b << " " << variance_b << " " << mean_b - width/2  << std::endl;
    std::cout << width << ", " << mean_a - width * p / (p + q)  << ", " << mean_b - width * p / (p + q) << std::endl;
    for (size_t j = 0; j < probs_a.size(); ++j) {
      std::cout << " " << j << ", " << exp(probs_a[j]) << ", " << exp(probs_b[j]) << std::endl;
    }
    double tv = GreedyMaxCoupling(probs_a, probs_b);
    //std::cout << width << " " << tv << std::endl;
  }
  
  /*for (size_t i = 1; i <= 4; ++i) {
    std::unordered_map<std::tuple<Bits, Bits, size_t>, double, tuple_hash> memoized;
    size_t width = exp2(ceil(log2(i)));
    Bits repr_a = (1 << i) - 1;
    Bits repr_b = 0;
    std::cout << i << " " << width << " " << repr_a << " " << repr_b << std::endl;
    std::unordered_map<Bits, double> parent_a = {{repr_a, log(1)}};
    std::unordered_map<Bits, double> parent_b = {{repr_b, log(1)}};
    auto probabilities_a = ComputeProbability(parent_a, memoized, 2*width, 2, values);
    auto probabilities_b = ComputeProbability(parent_b, memoized, 2*width, 2, values);
    for (auto [repr, value] : probabilities_a) {
      std::cout << repr << " " << exp(value) << std::endl;
    }
    for (auto [repr, value] : probabilities_b) {
      std::cout << repr << " " << exp(value) << std::endl;
    }
    double coupling = GreedyMaxCoupling(probabilities_a, probabilities_b);
    std::cout << i << " " << coupling << std::endl;
  }*/
  return 0;
}

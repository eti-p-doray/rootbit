from collections import defaultdict
from typing import List, Dict, Tuple
import math

import numpy as np
import utils
import numpy.typing as npt

def logmul(lhs: float, rhs: float) -> float:
  return lhs + rhs

def logadd(lhs: float, rhs: float) -> float:
  if lhs is None: return rhs
  if rhs is None: return lhs
  c = max(lhs, rhs)
  d = abs(lhs - rhs)
  return c + math.log1p(math.exp(-d))

def logsub(lhs: float, rhs: float) -> float:
  d = abs(lhs - rhs)
  if d < 1e-10: return None
  #print(lhs, rhs)
  return math.log(math.exp(lhs) - math.exp(rhs))

_factorial_table = [0]
for i in range(1, 65536):
  _factorial_table.append(_factorial_table[-1] + math.log(i))

def logfactorial(n: int) -> float:
  return _factorial_table[n]

def ComputeConditionalProbability(memoized_conditional: Dict[Tuple[str, str], float], width: int, symmetry_width: int, repr: str, parent_repr: str, values: List[float]) -> float:
  if (width == 2):
    # 1-p, p, q, 1-q
    if parent_repr == '0':
      if repr == '00':
        return logmul(values[0], values[0]) #one_hot((2, 0, 0, 0))
      if repr == '11':
        return logmul(values[1], values[1]) # one_hot((0, 2, 0, 0))
      return logmul(values[0], values[1]) # one_hot((1, 1, 0, 0))
    else:
      if repr == '00':
        return logmul(values[2], values[2]) # one_hot((0, 0, 2, 0))
      if repr == '11':
        return logmul(values[3], values[3]) # one_hot((0, 0, 0, 2))
      return logmul(values[2], values[3]) # one_hot((0, 0, 1, 1))

  if (repr, parent_repr) in memoized_conditional:
    return memoized_conditional[(repr, parent_repr)]

  half_width = width // 2

  left_parent = parent_repr[0:width // 4]
  right_parent = parent_repr[width // 4:]

  left_repr = repr[0:half_width]
  right_repr = repr[half_width:]

  probability_left = ComputeConditionalProbability(memoized_conditional, half_width, symmetry_width, left_repr, left_parent, values)
  probability_right = ComputeConditionalProbability(memoized_conditional, half_width, symmetry_width, right_repr, right_parent, values)
  probability = logmul(probability_left, probability_right)

  if (left_parent != right_parent and width <= symmetry_width):
    cross_probability_left = ComputeConditionalProbability(memoized_conditional, half_width, symmetry_width, left_repr, right_parent, values)
    cross_probability_right = ComputeConditionalProbability(memoized_conditional, half_width, symmetry_width, right_repr, left_parent, values)
    cross_probability = logmul(cross_probability_left, cross_probability_right)
    probability = logadd(probability, cross_probability)
    probability = logmul(probability, math.log(0.5))

  memoized_conditional[(repr, parent_repr)] = probability
  return probability

def ComputeProbability(parent_probabilities: Dict[str, float], memoized_conditional: Dict[Tuple[str, str], float], width: int, symmetry_width: int, values: List[float]) -> Dict[str, float]:
  classes = utils.GenerateEquivalentClasses(width, symmetry_width)

  result = {}
  for repr in classes:
    total_probability = None
    for parent_repr, parent_probability in parent_probabilities.items():
      if parent_probability is None:
        continue
      conditional = ComputeConditionalProbability(memoized_conditional, width, symmetry_width, repr, parent_repr, values)
      probability = logmul(parent_probability, conditional)
      total_probability = logadd(total_probability, probability) if total_probability is not None else probability

    coefficient = utils.GetEquivalentClassSize(repr, width, symmetry_width)
    total_probability = logmul(total_probability, math.log(coefficient))
    result[repr] = total_probability
  return result


# 1-p, p, q, 1-q
def ComputeConditionalHammingProbabilityPart1(v: int, parent_weight: int, values: List[float]) -> float:
  n = 2 * parent_weight;
  return values[3] * v + (values[2]) * (n - v) + logfactorial(n) - logfactorial(v) - logfactorial(n - v);

def ComputeConditionalHammingProbabilityPart0(u: int, parent_weight: int, values: List[float]) -> float:
  n = 2 * parent_weight;
  return values[1] * u + (values[0]) * (n - u) + logfactorial(n) - logfactorial(u) - logfactorial(n - u);

def ComputeConditionalHammingProbability(width: int, weight: int, parent_weight: int, values: List[float]) -> float:
  total_probability = None;
  # The parent string looks like 00..011.1 with `parent_weight` 1s and `width-parent_weight` 0s.

  # (u,v) splits the hamming weight into weight coming from parent 0s and 1s respectively.
  # u + v = weight
  for u in range(max(weight-2*parent_weight, 0), min(weight, 2*(width - parent_weight)) + 1):
    v = weight - u;
    
    probability = ComputeConditionalHammingProbabilityPart1(v, parent_weight, values) + ComputeConditionalHammingProbabilityPart0(u, width-parent_weight, values)
    if total_probability is None:
      total_probability = probability;
    elif probability is not None:
      total_probability = logadd(total_probability, probability)
  return total_probability;

def ComputeHammingProbabilities(parents: List[float], width: int, values: List[float]):
  results = [None] * (width + 1)
  half_width = width // 2
  for j in range(0, len(parents)):
    if parents[j] is None:
      continue
    
    for i in range(0, width+1):
      probability = ComputeConditionalHammingProbability(half_width, i, j, values)
      probability += parents[j]
      if results[i] is None:
        results[i] = probability
      elif probability is not None:
        results[i] = logadd(results[i], probability)
  return results

def nplogadd(lhs: npt.NDArray, rhs: npt.NDArray) -> float:
  c = np.maximum(lhs, rhs)
  d = np.absolute(lhs - rhs)
  r = c + np.log1p(np.exp(-d))
  r[np.isinf(rhs)] = lhs[np.isinf(rhs)]
  r[np.isinf(lhs)] = rhs[np.isinf(lhs)]
  return r

def ComputeDynamicProbability(parents_0: npt.NDArray, parents_1: npt.NDArray, values):
  a = math.exp(values[0]) * parents_0 + math.exp(values[1]) * parents_1
  b = math.exp(values[2]) * parents_0 + math.exp(values[3]) * parents_1
  probs_0 = np.multiply.outer(a, a)[np.triu_indices(a.size)]
  probs_1 = np.multiply.outer(b, b)[np.triu_indices(a.size)]
  return probs_0, probs_1
    
def ComputeDynamicStrings(parent_strings: npt.NDArray):
  return np.add.outer(parent_strings, parent_strings).flatten()

def ComputeDynamicEquivalenceClassSizes(parent_sizes: npt.NDArray):
  return np.multiply(np.multiply.outer(parent_sizes, parent_sizes), 2*np.ones((parent_sizes.size, parent_sizes.size)) - np.identity(parent_sizes.size))[np.triu_indices(parent_sizes.size)]

def ComputeRecursiveProbability(memoized: Dict[Tuple[str, str], float], repr, root, width, values: List[float]):
  if width == 1:
    if root == '0':
      if repr == '0':
        return values[0]
      else:
        return values[1]
    if root == '1':
      if repr == '1':
        return values[3]
      else:
        return values[2]
  if (repr, root) in memoized:
    return memoized[(repr, root)]

  half_width = width // 2
  left_repr = repr[0:half_width]
  right_repr = repr[half_width:]
  if root == '0':
    prob = logadd(
      logmul(values[0], logmul(
        ComputeRecursiveProbability(memoized, left_repr, '0', half_width, values),
        ComputeRecursiveProbability(memoized, right_repr, '0', half_width, values))), 
      logmul(values[1], logmul(
        ComputeRecursiveProbability(memoized, left_repr, '1', half_width, values),
        ComputeRecursiveProbability(memoized, right_repr, '1', half_width, values))))
  else:
    prob = logadd(
      logmul(values[2], logmul(
        ComputeRecursiveProbability(memoized, left_repr, '0', half_width, values),
        ComputeRecursiveProbability(memoized, right_repr, '0', half_width, values))), 
      logmul(values[3], logmul(
        ComputeRecursiveProbability(memoized, left_repr, '1', half_width, values),
        ComputeRecursiveProbability(memoized, right_repr, '1', half_width, values))))
  memoized[(repr, root)] = prob
  return prob

def mean(values: List[float]):
  if len(values) <= 1:
    return 0.0
  mean = 0.0
  for i in range(0, len(values)):
    mean += i * math.exp(values[i])
  return mean


def variance(values: List[float]):
  if len(values) <= 1:
    return 0.0
  m = mean(values)
  v = 0.0
  for i in range(0, len(values)):
    v += (i-m) * (i-m) * math.exp(values[i])
  return v
from collections import defaultdict
from typing import List, Dict, Tuple
import math

import utils

def logmul(lhs: float, rhs: float) -> float:
  return lhs + rhs

def logadd(lhs: float, rhs: float) -> float:
  c = min(lhs, rhs)
  return c + math.log(math.exp(lhs - c) + math.exp(rhs - c))

def logsub(lhs: float, rhs: float) -> float:
  c = min(lhs, rhs)
  return c + math.log(math.exp(lhs - c) - math.exp(rhs - c))

def ComputeConditionalProbability(memoized_conditional: Dict[Tuple[str, str], float], width: int, repr: str, parent_repr: str, values: List[float]) -> float:
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

  probability_left = ComputeConditionalProbability(memoized_conditional, half_width, left_repr, left_parent, values)
  probability_right = ComputeConditionalProbability(memoized_conditional, half_width, right_repr, right_parent, values)
  probability = logmul(probability_left, probability_right)

  if (left_parent != right_parent):
    cross_probability_left = ComputeConditionalProbability(memoized_conditional, half_width, left_repr, right_parent, values)
    cross_probability_right = ComputeConditionalProbability(memoized_conditional, half_width, right_repr, left_parent, values)
    cross_probability = logmul(cross_probability_left, cross_probability_right)
    probability = logmul(logadd(probability, cross_probability), math.log(0.5))

  memoized_conditional[(repr, parent_repr)] = probability
  return probability

def ComputeProbability(parent_probabilities: Dict[str, float], memoized_conditional: Dict[Tuple[str, str], float], width: int, values: List[float]) -> Dict[str, float]:
  half_width = width // 2
  classes = utils.GenerateEquivalentClasses(width)

  result = {}
  for repr in classes:
    total_probability = None
    for parent_repr, parent_probability in parent_probabilities.items():
      if parent_probability is None:
        continue
      conditional = ComputeConditionalProbability(memoized_conditional, width, repr, parent_repr, values)
      probability = logmul(parent_probability, conditional)
      total_probability = logadd(total_probability, probability) if total_probability is not None else probability

    coefficient = utils.GetEquivalentClassSize(repr, width)
    total_probability = logmul(total_probability, math.log(coefficient))
    result[repr] = total_probability
  return result
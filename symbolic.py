import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
import math

import utils
import numerical

PolynomialTerm = Tuple[int, int, int, int]
Polynomial = Dict[PolynomialTerm, int]

def polymul(lhs: Polynomial, rhs: Polynomial) -> Polynomial:
  result = defaultdict(int)
  for i, a in lhs.items():
    for j, b in rhs.items():
      index = tuple(np.array(i) + np.array(j))
      result[index] += a * b
  return result

def scalarmul(poly: Polynomial, scalar: float) -> Polynomial:
  result = poly.copy()
  for i in result:
    result[i] *= scalar
  return result

def polyadd(lhs: Polynomial, rhs: Polynomial) -> Polynomial:
  result = defaultdict(int)
  for i, a in lhs.items():
    result[i] += a
  for j, b in rhs.items():
    result[j] += b
  return result

def polysub(lhs: Polynomial, rhs: Polynomial) -> Polynomial:
  result = defaultdict(int)
  for i, a in lhs.items():
    result[i] -= a
  for j, b in rhs.items():
    result[j] -= b
  return result

def polyeval(poly: Polynomial, values: List[float]) -> Polynomial:
  result = None
  for i, a in poly.items():
    term = math.log(a)
    for k, j in enumerate(i):
      term = numerical.logmul(term, (values[k] * j))
    if result is None:
      result = term
    else:
      result = numerical.logadd(result, term)
  return result

def one_hot(index: Polynomial, coefficient: int = 1) -> Polynomial:
  result = defaultdict(int)
  result[index] = coefficient
  return result

def ComputeConditionalProbability(memoized_conditional: Dict[Tuple[str, str], Polynomial], width: int, repr: str, parent_repr: str) -> Polynomial:
  if (width == 1):
    if parent_repr == '0':
      if repr == '0':
        return one_hot((1, 0, 0, 0))
      else:
        return one_hot((0, 1, 0, 0))
    else:
      if repr == '0':
        return one_hot((0, 0, 1, 0))
      else:
        return one_hot((0, 0, 0, 1))
  if (width == 2):
    # 1-p, p, q, 1-q
    if parent_repr == '0':
      if repr == '00':
        return one_hot((2, 0, 0, 0))
      if repr == '11':
        return one_hot((0, 2, 0, 0))
      return one_hot((1, 1, 0, 0))
    else:
      if repr == '00':
        return one_hot((0, 0, 2, 0))
      if repr == '11':
        return one_hot((0, 0, 0, 2))
      return one_hot((0, 0, 1, 1))

  if (repr, parent_repr) in memoized_conditional:
    return memoized_conditional[(repr, parent_repr)]

  half_width = width // 2

  left_parent = parent_repr[0:width // 4]
  right_parent = parent_repr[width // 4:]

  left_repr = repr[0:half_width]
  right_repr = repr[half_width:]

  probability_left = ComputeConditionalProbability(memoized_conditional, half_width, left_repr, left_parent)
  probability_right = ComputeConditionalProbability(memoized_conditional, half_width, right_repr, right_parent)
  probability = polymul(probability_left, probability_right)

  if (left_parent != right_parent):
    cross_probability_left = ComputeConditionalProbability(memoized_conditional, half_width, left_repr, right_parent)
    cross_probability_right = ComputeConditionalProbability(memoized_conditional, half_width, right_repr, left_parent)
    cross_probability = polymul(cross_probability_left, cross_probability_right)
    probability = polyadd(probability, cross_probability)

  memoized_conditional[(repr, parent_repr)] = probability
  return probability


def ComputeProbability(parent_probabilities: Dict[str, Polynomial], memoized_conditional: Dict[Tuple[str, str], Polynomial], width: int) -> Dict[str, Polynomial]:
  half_width = width // 2
  classes = utils.GenerateEquivalentClasses(width, width)

  result = {}
  for repr in classes:
    total_probability = defaultdict(int)
    for parent_repr, parent_probability in parent_probabilities.items():
      conditional = ComputeConditionalProbability(memoized_conditional, width, repr, parent_repr)
      probability = polymul(parent_probability, conditional)
      total_probability = polyadd(total_probability, probability)
    #coefficient = utils.GetEquivalentClassSize(repr, width)
    #total_probability = scalarmul(total_probability, coefficient)
    result[repr] = total_probability
  return result

def polynomial_to_str(poly: Polynomial) -> str:
  lookup = ['a', 'b', 'c', 'd']
  #lookup = ['(1-p)', 'p', 'q', '(1-q)']
  result = ''
  for idx, a in poly.items():
    result += '+' + str(a) + '('
    result += '*'.join([lookup[k] + '^' + str(i) for k, i in filter(lambda i: i[1] > 0, enumerate(idx))])
    result += ') '
  return result

def ComputeRecursiveProbability(repr, root, width):
  if width == 1:
    if root == '0':
      if repr == '0':
        return one_hot((1, 0, 0, 0))
      else:
        return one_hot((0, 1, 0, 0))
    if root == '1':
      if repr == '1':
        return one_hot((0, 0, 0, 1))
      else:
        return one_hot((0, 0, 1, 0))

  half_width = width // 2
  left_repr = repr[0:half_width]
  right_repr = repr[half_width:]
  if root == '0':
    prob = polyadd(
      polymul(one_hot((1, 0, 0, 0)), polymul(
        ComputeRecursiveProbability(left_repr, '0', half_width),
        ComputeRecursiveProbability(right_repr, '0', half_width))), 
      polymul(one_hot((0, 1, 0, 0)), polymul(
        ComputeRecursiveProbability(left_repr, '1', half_width),
        ComputeRecursiveProbability(right_repr, '1', half_width))))
    return prob
  else:
    prob = polyadd(
      polymul(one_hot((0, 0, 1, 0)), polymul(
        ComputeRecursiveProbability(left_repr, '0', half_width),
        ComputeRecursiveProbability(right_repr, '0', half_width))), 
      polymul(one_hot((0, 0, 0, 1)), polymul(
        ComputeRecursiveProbability(left_repr, '1', half_width),
        ComputeRecursiveProbability(right_repr, '1', half_width))))
    return prob
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os

def get_bin(value, width) -> str: 
  return format(value, 'b').zfill(width)

def GenerateEquivalentClassesImpl(table: Dict[Tuple[int, int], List[str]], sum: int, width: int) -> List[str]:
  if (width == 1):
    return [get_bin(sum, width)]
  if (sum, width) in table:
    return table[(sum, width)]
  result = []
  half_width = width // 2
  for i in range(0, sum // 2 + 1):
    if i > sum - i:
      break
    if (sum - i) > half_width:
      continue
    if i > half_width:
      continue
    if (i == sum - i):
      left_list = GenerateEquivalentClassesImpl(table, i, half_width)
      for j in range(len(left_list)):
        for k in range(j+1):
          result.append(left_list[j] + left_list[k])
    else:
      left_list = GenerateEquivalentClassesImpl(table, sum - i, half_width)
      right_list = GenerateEquivalentClassesImpl(table, i, half_width)
      for left_entry in left_list:
        for right_entry in right_list:
          result.append(left_entry + right_entry)
  table[(sum, width)] = result
  return result

def GenerateEquivalentClasses(width: int) -> List[str]:
  result = []
  table = {}
  for i in range(0, width+1):
    result.extend(GenerateEquivalentClassesImpl(table, i, width))
  return result

def GetEquivalentClassSize(value: str, width: int) -> int:
  if (width == 1):
    return 1
  half_width = width // 2
  left = GetEquivalentClassSize(value[0:half_width], half_width)
  if (value[0:half_width] != value[half_width:]):
    right = GetEquivalentClassSize(value[half_width:], half_width)
    return left * right * 2
  return left * left

width = 4
classes = GenerateEquivalentClasses(width)
for value in classes:
  print(value, GetEquivalentClassSize(value, width))

PolynomialTerm = Tuple[int, int, int, int]
Polynomial = Dict[PolynomialTerm, int]

def polymul(lhs: Polynomial, rhs: Polynomial) -> Polynomial:
  result = defaultdict(int)
  for i, a in lhs.items():
    for j, b in rhs.items():
      index = tuple(np.array(i) + np.array(j))
      result[index] += a * b
  return result

def scalarmul(poly: Polynomial, scalar: int) -> Polynomial:
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

def polyeval(poly: Polynomial, values: List[float]) -> Polynomial:
  result = 0.0
  for i, a in poly.items():
    term = a
    for k, j in enumerate(i):
      term *= (values[k] ** j)
    result += term
  return result

def one_hot(index: Polynomial, coefficient: int = 1) -> Polynomial:
  result = defaultdict(int)
  result[index] = coefficient
  return result

def ComputeSymbolicConditionalProbability(memoized_conditional: Dict[Tuple[str, str], Polynomial], width: int, repr: str, parent_repr: str) -> Polynomial:
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

  probability_left = ComputeSymbolicConditionalProbability(memoized_conditional, half_width, left_repr, left_parent)
  probability_right = ComputeSymbolicConditionalProbability(memoized_conditional, half_width, right_repr, right_parent)
  probability = polymul(probability_left, probability_right)

  if (left_parent != right_parent):
    cross_probability_left = ComputeSymbolicConditionalProbability(memoized_conditional, half_width, left_repr, right_parent)
    cross_probability_right = ComputeSymbolicConditionalProbability(memoized_conditional, half_width, right_repr, left_parent)
    cross_probability = polymul(cross_probability_left, cross_probability_right)
    probability = polyadd(probability, cross_probability)

  memoized_conditional[(repr, parent_repr)] = probability
  return probability


def ComputeSymbolicProbability(parent_probabilities: Dict[str, Polynomial], memoized_conditional: Dict[Tuple[str, str], Polynomial], width: int) -> Dict[str, Polynomial]:
  half_width = width // 2
  classes = GenerateEquivalentClasses(width)

  result = {}
  for repr in classes:
    total_probability = defaultdict(int)
    for parent_repr, parent_probability in parent_probabilities.items():
      conditional = ComputeSymbolicConditionalProbability(memoized_conditional, width, repr, parent_repr)
      probability = polymul(parent_probability, conditional)
      total_probability = polyadd(total_probability, probability)
    coefficient = GetEquivalentClassSize(repr, width)
    total_probability = scalarmul(total_probability, coefficient)
    result[repr] = total_probability
    #print(repr, polynomial_to_str(total_probability))
  return result

def ComputeNumericalConditionalProbability(memoized_conditional: Dict[Tuple[str, str], float], width: int, repr: str, parent_repr: str, values: List[float]) -> float:
  if (width == 2):
    # 1-p, p, q, 1-q
    if parent_repr == '0':
      if repr == '00':
        return values[0] * values[0] #one_hot((2, 0, 0, 0))
      if repr == '11':
        return values[1] * values[1] # one_hot((0, 2, 0, 0))
      return values[0] * values[1] # one_hot((1, 1, 0, 0))
    else:
      if repr == '00':
        return values[2] * values[2] # one_hot((0, 0, 2, 0))
      if repr == '11':
        return values[3] * values[3] # one_hot((0, 0, 0, 2))
      return values[2] * values[3] # one_hot((0, 0, 1, 1))

  if (repr, parent_repr) in memoized_conditional:
    return memoized_conditional[(repr, parent_repr)]

  half_width = width // 2

  left_parent = parent_repr[0:width // 4]
  right_parent = parent_repr[width // 4:]

  left_repr = repr[0:half_width]
  right_repr = repr[half_width:]

  probability_left = ComputeNumericalConditionalProbability(memoized_conditional, half_width, left_repr, left_parent, values)
  probability_right = ComputeNumericalConditionalProbability(memoized_conditional, half_width, right_repr, right_parent, values)
  probability = probability_left * probability_right

  if (left_parent != right_parent):
    cross_probability_left = ComputeNumericalConditionalProbability(memoized_conditional, half_width, left_repr, right_parent, values)
    cross_probability_right = ComputeNumericalConditionalProbability(memoized_conditional, half_width, right_repr, left_parent, values)
    cross_probability = cross_probability_left * cross_probability_right
    probability = (probability + cross_probability)/2

  memoized_conditional[(repr, parent_repr)] = probability
  return probability

def ComputeNumericalProbability(parent_probabilities: Dict[str, float], memoized_conditional: Dict[Tuple[str, str], float], width: int, values: List[float]) -> Dict[str, float]:
  half_width = width // 2
  classes = GenerateEquivalentClasses(width)
  print(len(classes))

  result = {}
  for repr in classes:
    total_probability = 0.0
    for parent_repr, parent_probability in parent_probabilities.items():
      conditional = ComputeNumericalConditionalProbability(memoized_conditional, width, repr, parent_repr, values)
      probability = parent_probability * conditional
      total_probability = total_probability + probability
    coefficient = GetEquivalentClassSize(repr, width)
    total_probability = total_probability * coefficient
    result[repr] = total_probability
  return result

def polynomial_to_str(poly: Polynomial) -> str:
  lookup = ['(1-p)', 'p', 'q', '(1-q)']
  result = ''
  for idx, a in poly.items():
    result += '+' + str(a) + '('
    for k, i in enumerate(idx):
      result += str(i) + ','
    result += ') '
  return result

# Ensure the directory exists, if not, create it
output_directory = "./new_root_bit_plots"
if not os.path.exists(output_directory):
  os.makedirs(output_directory)


bar_width = 3.0
spacing = 8.4 # spacing between bar groups

p = 0.1
q = p
values1 = [1.0 - p, p, q, 1.0-q]
memoized_conditional_0 = {}
memoized_conditional_1 = {}
probabilities_0 = {'0': 1.0, '1': 0.0}
probabilities_1 = {'0': 0.0, '1': 1.0}
for depth in range(1, 6):
  width = 2 ** depth
  print(width)
  probabilities_0 = ComputeNumericalProbability(probabilities_0, memoized_conditional_0, width, values1)
  probabilities_1 = ComputeNumericalProbability(probabilities_1, memoized_conditional_1, width, values1)

  print(probabilities_0)
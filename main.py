import matplotlib.pyplot as plt
import os
import math

import numerical

def bit_diff(lhs, rhs):
  return sum(c1!=c2 for c1,c2 in zip(lhs,rhs))

def AverageCoupledBits(probabilities_0, probabilities_1):
  i, j = 0, 0
  coupling_structure = []

  keys_0 = sorted(probabilities_0.keys(), key=lambda repr: repr.count('1'))
  keys_1 = sorted(probabilities_1.keys(), key=lambda repr: repr.count('1'))

  weight = bit_diff(keys_0[0],keys_1[0])
  sum_0 = probabilities_0[keys_0[0]]
  sum_1 = probabilities_1[keys_1[0]]
  overlap = min(sum_0, sum_1)
  coupling_sum = numerical.logmul(overlap, math.log(weight)) if weight != 0 else None
  coupling_structure.append({
    '0': keys_0[0],
    '1': keys_1[0],
    'overlap': math.exp(overlap),
    'weight': weight,
    'start': 0.0,
  })

  if sum_0 < sum_1:
    i += 1
  else:
    j += 1

  while i < len(keys_0) and j < len(keys_1):
    weight = bit_diff(keys_0[i],keys_1[j])
    start = min(math.exp(sum_0), math.exp(sum_1))
    if sum_0 < sum_1:
      prob = probabilities_0[keys_0[i]]
      new_sum_0 = numerical.logadd(sum_0, prob)
      overlap = prob if new_sum_0 < sum_1 else numerical.logsub(sum_1, sum_0)
      sum_0 = new_sum_0
    else:
      prob = probabilities_1[keys_1[j]]
      new_sum_1 = numerical.logadd(sum_1, prob)
      overlap = prob if new_sum_1 < sum_0 else numerical.logsub(sum_0, sum_1)
      sum_1 = new_sum_1
    new_coupling = numerical.logmul(overlap, math.log(weight)) if weight != 0 else None

    coupling_structure.append({
      '0': keys_0[i],
      '1': keys_1[j],
      'overlap': math.exp(overlap),
      'weight': weight,
      'start': start,
    })

    if sum_0 < sum_1:
      i += 1
    else:
      j += 1
    
    if new_coupling is not None:
      coupling_sum = numerical.logadd(coupling_sum, new_coupling) if coupling_sum is not None else new_coupling
    

  return coupling_sum, coupling_structure




p = 0.1
q = p
values = [math.log(1.0 - p), math.log(p), math.log(q), math.log(1.0-q)]
memoized_conditional_0 = {}
memoized_conditional_1 = {}
probabilities_0 = {'0': math.log(1.0), '1': None}
probabilities_1 = {'0': None, '1': math.log(1.0)}
for depth in range(1, 3):
  width = 2 ** depth
  print(width)
  probabilities_0 = numerical.ComputeProbability(probabilities_0, memoized_conditional_0, width, values)
  probabilities_1 = numerical.ComputeProbability(probabilities_1, memoized_conditional_1, width, values)

  #print(probabilities_0)
  total = None
  for key, prob in probabilities_0.items():
    #print(key, prob, total)
    total = numerical.logadd(total, prob) if total is not None else prob

  coupling_sum, coupling_structure = AverageCoupledBits(probabilities_0, probabilities_1)
  print('coupling: ', math.exp(coupling_sum))
  for coupling in coupling_structure:
    print(coupling)

  #print(sorted(probabilities_0.keys(), key=lambda repr: repr.count('1')))
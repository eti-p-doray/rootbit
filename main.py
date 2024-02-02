import matplotlib.pyplot as plt
import os
import math
from typing import List, Dict, Tuple
import scipy
import numpy as np
import random
from copy import deepcopy

import numerical
import utils

def hamming_distance(table: Dict[Tuple[str, str], int], lhs, rhs, symmetry_width):
  if (lhs, rhs) in table:
    return table[(lhs, rhs)]
  width = len(lhs)
  if width == 1:
    return int(lhs != rhs)
  half_width = width // 2
  d1 = hamming_distance(table, lhs[0:half_width], rhs[0:half_width], symmetry_width) + hamming_distance(table, lhs[half_width:], rhs[half_width:], symmetry_width)
  if width > symmetry_width:
    return d1
  d2 = hamming_distance(table, lhs[0:half_width], rhs[half_width:], symmetry_width) + hamming_distance(table, lhs[half_width:], rhs[0:half_width], symmetry_width)
  d = min(d1, d2)
  table[(lhs, rhs)] = d
  return d

def GreedyMaxCoupling(probabilities_a, probabilities_b):
  #probabilities_a = deepcopy(probabilities_a)
  #probabilities_b = deepcopy(probabilities_b)
  keys_0 = [key for key in probabilities_a.keys()]
  keys_1 = [key for key in probabilities_b.keys()]
  hamming_table = {}

  coupling_structure = []

  sum_overlap = None
  overlap = None
  for key in keys_0:
    if probabilities_a[key] is None or probabilities_b[key] is None:
      continue
    prob_a = probabilities_a[key]
    prob_b = probabilities_b[key]
    overlap = min(prob_a, prob_b)
    start = math.exp(sum_overlap) if sum_overlap is not None else 0.0
    sum_overlap = numerical.logadd(sum_overlap, overlap) if sum_overlap is not None else overlap
    #probabilities_a[key] = numerical.logsub(prob_a, overlap) if prob_a > overlap else None
    #probabilities_b[key] = numerical.logsub(prob_b, overlap) if prob_b > overlap else None
    coupling_structure.append({
      'a': key,
      'b': key,
      'overlap': math.exp(overlap),
      'weight': 0,
      'start': start
    })

  return math.exp(sum_overlap), coupling_structure

def LPOptimalCoupling(probabilities_a, probabilities_b, symmetry_width):
  coupling_structure = []
  hamming_table = {}

  keys_0 = [key for key in probabilities_a.keys()]
  keys_1 = [key for key in probabilities_b.keys()]

  obj_c = np.zeros(len(probabilities_a) * len(probabilities_b))
  A_eq = np.zeros((len(probabilities_a) + len(probabilities_b), len(probabilities_a) * len(probabilities_b)))
  b_eq = np.zeros(len(probabilities_a) + len(probabilities_b))

  for i, key_0 in enumerate(keys_0):
    for j, key_1 in enumerate(keys_1):
      d = hamming_distance(hamming_table, key_0, key_1, symmetry_width)
      obj_c[i * len(probabilities_b) + j] = d + (1.0 if d > 0 else 0)
      A_eq[i, i * len(probabilities_b) + j] = 1
      A_eq[j + len(probabilities_a), i * len(probabilities_b) + j] = 1
  for i, key_0 in enumerate(keys_0):
    b_eq[i] = np.exp(probabilities_a[key_0])
  for i, key_1 in enumerate(keys_1):
    b_eq[i + len(probabilities_a)] = np.exp(probabilities_b[key_1])

  solution = scipy.optimize.linprog(obj_c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None))
  sol_x, coupling = solution.x, solution.fun

  sum_overlap = 0
  avg_coupling = 0
  sum_coupling = 0
  for i, key_0 in enumerate(keys_0):
    for j, key_1 in enumerate(keys_1):
      overlap = sol_x[i * len(probabilities_b) + j]
      if overlap == 0:
        continue
      d = hamming_distance(hamming_table, key_0, key_1, symmetry_width)
      avg_coupling += overlap * d
      if d == 0:
        sum_coupling += overlap
      coupling_structure.append({
        'a': key_0,
        'b': key_1,
        'overlap': overlap,
        'weight': d,
        'start': sum_overlap,
      })
      sum_overlap += overlap

  return sum_coupling, avg_coupling, coupling_structure

def NaiveCoupling(probabilities_a, probabilities_b):
  keys_0 = [key for key in probabilities_a.keys()]
  keys_1 = [key for key in probabilities_b.keys()]
  hamming_table = {}

  i, j = 0, 0
  coupling_structure = []
  sum_0 = probabilities_a[keys_0[0]]
  sum_1 = probabilities_b[keys_1[0]]
  overlap = min(sum_0, sum_1)
  coupling_structure.append({
    'a': keys_0[0],
    'b': keys_1[0],
    'overlap': math.exp(overlap),
    'weight': 0,
    'start': 0.0,
  })
  coupling_sum = None

  if sum_0 < sum_1:
    i += 1
  else:
    j += 1

  while i < len(keys_0) and j < len(keys_1):
    print(keys_0[i],keys_1[j])
    weight = hamming_distance(hamming_table, keys_0[i],keys_1[j])
    start = min(math.exp(sum_0), math.exp(sum_1))
    if sum_0 < sum_1:
      prob = probabilities_a[keys_0[i]]
      new_sum_0 = numerical.logadd(sum_0, prob)
      overlap = prob if new_sum_0 < sum_1 else numerical.logsub(sum_1, sum_0)
      sum_0 = new_sum_0
    else:
      prob = probabilities_b[keys_1[j]]
      new_sum_1 = numerical.logadd(sum_1, prob)
      overlap = prob if new_sum_1 < sum_0 else numerical.logsub(sum_0, sum_1)
      sum_1 = new_sum_1
    new_coupling = numerical.logmul(overlap, math.log(weight)) if weight != 0 else None

    coupling_structure.append({
      'a': keys_0[i],
      'b': keys_1[j],
      'overlap': math.exp(overlap),
      'weight': weight,
      'start': start,
    })

    if sum_0 < sum_1:
      i += 1
    else:
      j += 1
    
    if new_coupling is not None:
      coupling_sum = numerical.logadd(coupling_sum, new_coupling) if coupling_sum else new_coupling

  return math.exp(coupling_sum), coupling_structure


import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

def plotCoupling(coupling_structure, group_names, title, avg_coupling, sum_coupling):
  fig = make_subplots(rows=3, cols=1, subplot_titles=(group_names['a'], group_names['b'], f'Weight: {avg_coupling} - {sum_coupling}')) 
  fig.update_xaxes(range=[0, 1.0])

  x = [coupling['start']+coupling['overlap']/2 for coupling in coupling_structure]
  overlap = [coupling['overlap'] for coupling in coupling_structure]
  weight=[coupling['weight'] for coupling in coupling_structure]
  fig.append_trace(go.Bar(x=x, 
                          y=[1 for _ in coupling_structure],
                          width=overlap, 
                          hovertemplate = 'Width: %{width:.4f}',
                          marker_color=[px.colors.qualitative.Plotly[int(coupling['a'], base=2) % 10] for coupling in coupling_structure],
                          text=[coupling['a'] for coupling in coupling_structure], 
                          name=group_names['a']), row=1, col=1)
  fig.append_trace(go.Bar(x=x, 
                          y=[1 for _ in coupling_structure], 
                          width=overlap, 
                          hovertemplate = 'Width: %{width:.4f}',
                          marker_color=[px.colors.qualitative.Plotly[int(coupling['b'], base=2) % 10] for coupling in coupling_structure],
                          text=[coupling['b'] for coupling in coupling_structure],
                          name=group_names['b']), row=2, col=1)  
  fig.append_trace(go.Bar(x=x, y=weight, width=overlap, name='Weight', hovertemplate = 'Width: %{width:.4f} Weight: %{y}'), row=3, col=1)
  fig.update_layout(title=title, yaxis=dict(
        showticklabels=False
    ), yaxis2=dict(
        showticklabels=False
    ), yaxis3=dict(
        title="Weight",
    ),)
  fig.update_traces(showlegend=False)

  fig.show()
  fig.write_image(f"results/coupling_{title}.jpg")


pd.set_option('display.max_rows', None)
p = 0.2
q = 0.05
values = [math.log(1.0 - p), math.log(p), math.log(q), math.log(1.0-q)]
memoized_conditional = {}


probabilities_0 = {'0': math.log(1.0)}
probabilities_1 = {'1': math.log(1.0)}


for depth in range(1,5):
  width = 2 ** depth
  print(width)
  probabilities_0 = numerical.ComputeProbability(probabilities_0, memoized_conditional, width, width, values)
  probabilities_1 = numerical.ComputeProbability(probabilities_1, memoized_conditional, width, width, values)
  #sum_coupling, coupling_structure = GreedyMaxCoupling(probabilities_0, probabilities_1)
  sum_coupling, avg_coupling, coupling_structure = LPOptimalCoupling(probabilities_0, probabilities_1, width)
  print(f'1-{width}: ', sum_coupling, avg_coupling)
  print(len(probabilities_0))
  #plotCoupling(coupling_structure, {'a': 'Root 0', 'b': 'Root 1'}, f'Coupling-1-{width}', avg_coupling, sum_coupling)

"""for bits in range(2,17):
  memoized_conditional = {}
  width = 2 ** math.ceil(math.log(bits, 2))
  print(bits, width)
  group_a = '0' * width
  group_b = '1' * bits + '0' * (width - bits)
  probabilities_0 = numerical.ComputeProbability({group_a: math.log(1.0)}, memoized_conditional, 2*width, values)
  #print(probabilities_0)
  probabilities_1 = numerical.ComputeProbability({group_b: math.log(1.0)}, memoized_conditional, 2*width, values)

  sum_coupling, coupling_structure = GreedyMaxCoupling(probabilities_0, probabilities_1)
  print(f'Coupling {bits}-{2*bits}: ', sum_coupling)
  #sum_coupling, avg_coupling, coupling_structure = LPOptimalCoupling(probabilities_0, probabilities_1)
  #print(f'Coupling {bits}-{2*bits}: ', sum_coupling)
  #plotCoupling(coupling_structure, {'a': f'Parent {group_a}', 'b': f'Parent {group_b}'}, f'Coupling-{bits}-{2*bits}', avg_coupling, sum_coupling)
"""
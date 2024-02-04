import matplotlib.pyplot as plt
import os
import math
from typing import List, Dict, Tuple
import scipy
import numpy as np
import random
from copy import deepcopy
import operator

import numerical
import utils

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
      d = utils.HammingDistance(hamming_table, key_0, key_1, symmetry_width)
      w = abs(utils.HammingWeight(key_0) - utils.HammingWeight(key_1))
      obj_c[i * len(probabilities_b) + j] = d + (1.0 if d > 0 else 0)
      if d != w:
        continue
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
      d = utils.HammingDistance(hamming_table, key_0, key_1, symmetry_width)
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

def LPOptimalHammingCoupling(probabilities_a, probabilities_b):
  coupling_structure = []
  hamming_table = {}

  obj_c = np.zeros(len(probabilities_a) * len(probabilities_b))
  A_eq = np.zeros((len(probabilities_a) + len(probabilities_b), len(probabilities_a) * len(probabilities_b)))
  b_eq = np.zeros(len(probabilities_a) + len(probabilities_b))

  for i, v_a in enumerate(probabilities_a):
    for j, v_b in enumerate(probabilities_b):
      obj_c[i * len(probabilities_b) + j] = abs(i-j) + (1.0 if i!=j else 0)
      A_eq[i, i * len(probabilities_b) + j] = 1
      A_eq[j + len(probabilities_a), i * len(probabilities_b) + j] = 1
  for i, v_a in enumerate(probabilities_a):
    b_eq[i] = np.exp(v_a)
  for i, v_b in enumerate(probabilities_b):
    b_eq[i + len(probabilities_a)] = np.exp(v_b)

  solution = scipy.optimize.linprog(obj_c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None))
  sol_x, coupling = solution.x, solution.fun

  sum_overlap = 0
  avg_coupling = 0
  sum_coupling = 0
  for i, v_a in enumerate(probabilities_a):
    for j, v_b in enumerate(probabilities_b):
      d = abs(i-j)
      overlap = sol_x[i * len(probabilities_b) + j]
      if overlap == 0:
        continue
      avg_coupling += overlap * d
      if i == j:
        sum_coupling += overlap
      coupling_structure.append({
        'a': '0' + '1' * i,
        'b': '0' + '1' * j,
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
    weight = utils.HammingDistance(hamming_table, keys_0[i],keys_1[j])
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

"""
equiv_table = {}
hamming_table = {}
width = 16
previous_result = []
for i in range(0, width+1):
  print(i)
  result = utils.GenerateEquivalentClassesImpl(equiv_table, i, width, width)
  for j in result:
    for k in previous_result:
      if utils.HammingDistance(hamming_table, j, k, width) == 1:
        print(j, k)
  previous_result = result
"""


pd.set_option('display.max_rows', None)
p = 0.2 #( 1.0/2 - math.sqrt(1.0/8) )
q = 0.12 #p
values = [math.log(1.0 - p), math.log(p), math.log(q), math.log(1.0-q)]
memoized_conditional = {}


probabilities_0 = [math.log(1), None]
probabilities_1 = [None, math.log(1)]

for depth in range(1,3):
  width = 2 ** depth
  probabilities_0 = numerical.ComputeHammingProbabilities(probabilities_0, width, values)
  probabilities_1 = numerical.ComputeHammingProbabilities(probabilities_1, width, values)

  print(width, numerical.mean(probabilities_0), numerical.mean(probabilities_1), numerical.variance(probabilities_0), numerical.variance(probabilities_1))
  sum_coupling, avg_coupling, coupling_structure = LPOptimalHammingCoupling(probabilities_0, probabilities_1)
  print(numerical.mean(probabilities_1) - numerical.mean(probabilities_0))
  plotCoupling(coupling_structure, {'a': 'Root 0', 'b': 'Root 1'}, f'Coupling-1-{width}', avg_coupling, sum_coupling)


probabilities_0 = {'0': math.log(1.0)}
probabilities_1 = {'1': math.log(1.0)}
for depth in range(1,3):
  width = 2 ** depth
  print(width)
  probabilities_0 = numerical.ComputeProbability(probabilities_0, memoized_conditional, width, width, values)
  probabilities_1 = numerical.ComputeProbability(probabilities_1, memoized_conditional, width, width, values)
  #sum_coupling, coupling_structure = GreedyMaxCoupling(probabilities_0, probabilities_1)
  sum_coupling, avg_coupling, coupling_structure = LPOptimalCoupling(probabilities_0, probabilities_1, width)
  print(f'1-{width}: ', sum_coupling, avg_coupling)
  #print(len(probabilities_0))
  plotCoupling(coupling_structure, {'a': 'Root 0', 'b': 'Root 1'}, f'Coupling-1-{width}', avg_coupling, sum_coupling)

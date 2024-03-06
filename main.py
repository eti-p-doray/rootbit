import matplotlib.pyplot as plt
import os
import math
from typing import List, Dict, Tuple
import scipy
import numpy as np
import random
from copy import deepcopy
import operator
from queue import PriorityQueue
from pyvis.network import Network
from collections import defaultdict
import numpy as np

import numerical
import symbolic
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

def PathOptimalCoupling(probabilities_a, probabilities_b, symmetry_width):
  hamming_table = {}
  probabilities_a = deepcopy(probabilities_a)
  probabilities_b = deepcopy(probabilities_b)
  heap = PriorityQueue()
  for i, key_0 in enumerate(probabilities_a.keys()):
    for j, key_1 in enumerate(probabilities_b.keys()):
      d = utils.HammingDistance(hamming_table, key_0, key_1, symmetry_width)
      w = abs(utils.HammingWeight(key_0) - utils.HammingWeight(key_1))
      if d == w:
        heap.put((d, (key_0, key_1)))

  avg_coupling = 0
  sum_coupling = 0
  while not heap.empty():
    d, (key_0, key_1) = heap.get()
    prob_a = probabilities_a[key_0]
    prob_b = probabilities_b[key_1]
    if prob_a is None or prob_b is None:
      continue
    overlap = min(prob_a, prob_b)
    probabilities_a[key_0] = numerical.logsub(prob_a, overlap) if prob_a > overlap else None
    probabilities_b[key_1] = numerical.logsub(prob_b, overlap) if prob_b > overlap else None

    avg_coupling += math.exp(overlap) * d
    if d == 0:
      sum_coupling += math.exp(overlap)
  return sum_coupling, avg_coupling


def LPOptimalCoupling(probabilities_a, probabilities_b, symmetry_width, constrained):
  coupling_structure = []
  hamming_table = {}

  keys_0 = [key for key in probabilities_a.keys()]
  keys_1 = [key for key in probabilities_b.keys()]

  k = len(probabilities_a) * len(probabilities_b)
  if constrained:
    k = 0
    for i, key_0 in enumerate(keys_0):
      for j, key_1 in enumerate(keys_1):
        d = utils.HammingDistance(hamming_table, key_0, key_1, symmetry_width)
        w = abs(utils.HammingWeight(key_0) - utils.HammingWeight(key_1))
        if d == w:
          k += 1

  obj_c = np.zeros(k)
  A_eq = np.zeros((len(probabilities_a) + len(probabilities_b), k))
  b_eq = np.zeros(len(probabilities_a) + len(probabilities_b))

  k = 0
  for i, key_0 in enumerate(keys_0):
    for j, key_1 in enumerate(keys_1):
      d = utils.HammingDistance(hamming_table, key_0, key_1, symmetry_width)
      w = abs(utils.HammingWeight(key_0) - utils.HammingWeight(key_1))
      if d != w and constrained:
        continue
      obj_c[k] = d + (1.0 if d > 0 else 0)
      A_eq[i, k] = 1
      A_eq[j + len(probabilities_a), k] = 1
      k += 1
  for i, key_0 in enumerate(keys_0):
    b_eq[i] = np.exp(probabilities_a[key_0])
  for i, key_1 in enumerate(keys_1):
    b_eq[i + len(probabilities_a)] = np.exp(probabilities_b[key_1])

  solution = scipy.optimize.linprog(obj_c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None))
  sol_x, coupling = solution.x, solution.fun

  sum_overlap = 0
  avg_coupling = 0
  sum_coupling = 0
  k = 0
  for i, key_0 in enumerate(keys_0):
    for j, key_1 in enumerate(keys_1):
      d = utils.HammingDistance(hamming_table, key_0, key_1, symmetry_width)
      w = abs(utils.HammingWeight(key_0) - utils.HammingWeight(key_1))
      if d != w and constrained:
        continue
      overlap = sol_x[k]
      k += 1
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

def verifyPath(a, b, width, classes, hamming_table):
  if a == b:
    return True
  i = utils.HammingWeight(a)
  j = utils.HammingWeight(b)
  if i == j:
    return False
  for c in classes[i+1]:
    d = utils.HammingDistance(hamming_table, a, c, width)
    if d != 1:
      continue
    if c == b:
      return True
    if verifyPath(c, b, width, classes, hamming_table):
      return True
  return False

def verifyAllPaths(width):
  hamming_table = {}
  equiv_table = {}
  classes = []
  for i in range(0, width+1):
    classes.append(utils.GenerateEquivalentClassesImpl(equiv_table, i, width, width))
  for i in range(0, width+1):
    for j in range(i + 2, width+1):
      for a in classes[i]:
        for b in classes[j]:
          d = utils.HammingDistance(hamming_table, a, b, width)
          w = j - i
          if d != w:
            continue
          if not verifyPath(a, b, width, classes, hamming_table):
            print('Broken path: ', a, b)
          #else:
          #  print('Valid path: ', a, b)

def verifyFlowDirection(width, probabilities_a, probabilities_b):
  hamming_table = {}
  equiv_table = {}
  classes = []
  for i in range(0, width+1):
    classes.append(utils.GenerateEquivalentClassesImpl(equiv_table, i, width, width))
  for i in range(0, width):
    for a in classes[i]:
      #print(a, probabilities_a[a] > probabilities_b[a])
      if math.exp(probabilities_a[a]) > math.exp(probabilities_b[a]):
        continue
      for b in classes[i+1]:
        d = utils.HammingDistance(hamming_table, a, b, width)
        if d != 1:
          continue
        #print(a, b, probabilities_a[a] > probabilities_b[a], probabilities_a[b] > probabilities_b[b])
        if math.exp(probabilities_a[b]) > math.exp(probabilities_b[b]):
          print('Counter flow: ', a, b)

def plotNetwork(width, probabilities_a, probabilities_b):
  net = Network()

  hamming_table = {}
  equiv_table = {}
  classes = []
  for i in range(0, width+1):
    classes.append(utils.GenerateEquivalentClassesImpl(equiv_table, i, width, width))

  for i in range(0, width+1):
    for a in classes[i]:
      size = utils.GetEquivalentClassSize(a, width, width)
      prob_a = probabilities_a[a] - math.log(size)
      prob_b = probabilities_b[a] - math.log(size)
      value_min = min(prob_a, prob_b)
      value_max = max(prob_a, prob_b)
      net.add_node(a, color=f'rgb({255 * math.exp(prob_a - value_max)}, 0, {255 * math.exp(prob_b - value_max)})', x=i*200, fixed={'x':True}, label=a, title=f'a: {math.exp(prob_b)} b: {math.exp(prob_a)} d: {prob_a-prob_b}', value=math.exp(probabilities_a[a])+math.exp(probabilities_b[a]), scaling={'min':1})

  for i in range(0, width):
    for a in classes[i]:
      for b in classes[i+1]:
        d = utils.HammingDistance(hamming_table, a, b, width)
        if d == 1:
          net.add_edge(a, b, color='grey')

  """probabilities_a = deepcopy(probabilities_a)
  probabilities_b = deepcopy(probabilities_b)
  heap = PriorityQueue()
  for i, key_0 in enumerate(probabilities_a.keys()):
    for j, key_1 in enumerate(probabilities_b.keys()):
      d = utils.HammingDistance(hamming_table, key_0, key_1, width)
      w = abs(utils.HammingWeight(key_0) - utils.HammingWeight(key_1))
      if d == w:
        heap.put((d, (key_0, key_1)))

  while not heap.empty():
    d, (key_0, key_1) = heap.get()
    prob_a = probabilities_a[key_0]
    prob_b = probabilities_b[key_1]
    if prob_a is None or prob_b is None:
      continue
    overlap = min(prob_a, prob_b)
    probabilities_a[key_0] = numerical.logsub(prob_a, overlap) if prob_a > overlap else None
    probabilities_b[key_1] = numerical.logsub(prob_b, overlap) if prob_b > overlap else None

    if d > 0:
      net.add_edge(key_0, key_1, value=math.exp(overlap))"""

  net.repulsion()
  net.show_buttons(filter_=['physics'])
  net.show(f'results/network_{width}.html', notebook=False)


def mean(strings, probs, width):
  m = np.zeros(width)
  for i in range(0, strings.size):
    prob = math.exp(probs[i])
    value = np.array([int(j) for j in strings[i]])
    m += prob * value
  return m

def variance(strings, probs, width):
  m = mean(strings, probs, width)
  v = np.zeros((width, width))
  for i in range(0, strings.size):
    prob = math.exp(probs[i])
    value = np.array([int(j) for j in strings[i]])
    v += prob * np.outer(value - m, value - m)
  return v

def TotalVariation(probabilities_0, probabilities_1, equiv_size):
  tv = 0.0
  for i in range(0, probabilities_0.size):
    tv += min(probabilities_0[i], probabilities_1[i]) * equiv_size[i]
  return tv

def TotalVariationHamming(probabilities_0, probabilities_1):
  tv = 0.0
  for i in range(0, len(probabilities_0)):
    tv += math.exp(min(probabilities_0[i], probabilities_1[i]))
  return tv

def LowerBound(probabilities_0_in, probabilities_1_in, distance):

  previous_sum_overlap = 0.0
  cum_overlap = 0.0
  for k in  range(width, 0, -1):
    probabilities_0 = deepcopy(probabilities_0_in)
    probabilities_1 = deepcopy(probabilities_1_in)

    i = 0
    j = width
    sum_overlap = 0.0
    while i <= width and j >= 0  and i + k <= j:
      if probabilities_0[i] is None:
        i += 1
        continue
      if probabilities_1[j] is None:
        j -= 1
        continue
      #print(probabilities_0[i], probabilities_1[j])
      overlap = min(probabilities_0[i], probabilities_1[j])
      #print(i, j, probabilities_0[i], probabilities_1[j])
      sum_overlap += math.exp(overlap)
      #print(probabilities_0[i], probabilities_1[j], overlap)
      probabilities_0[i] = numerical.logsub(probabilities_0[i], overlap)
      probabilities_1[j] = numerical.logsub(probabilities_1[j], overlap)
      if probabilities_0[i] is None or probabilities_0[i] < probabilities_1[j]:
        i += 1
      else:
        j -= 1
    overlap = min(sum_overlap - previous_sum_overlap, distance / k)
    cum_overlap += overlap
    distance -= overlap * k
    print(k, sum_overlap, overlap, cum_overlap, distance)
    previous_sum_overlap = sum_overlap

pd.set_option('display.max_rows', None)
p = ( 1.0/2 - math.sqrt(1.0/8) ) - 0.0
q = ( 1.0/2 - math.sqrt(1.0/8) ) + 0.0
values = [math.log(1.0 - p), math.log(p), math.log(q), math.log(1.0-q)]

probabilities_0 = {'0': math.log(1.0)}
probabilities_1 = {'1': math.log(1.0)}
hamming_probabilities_0 = [math.log(1.0), None]
hamming_probabilities_1 = [None, math.log(1.0)]

distance_table = {}

probabilities_0 = np.array([1.0, 0.0])
probabilities_1 = np.array([0.0, 1.0])
#strings = np.array(['0', '1'], dtype=object)
equiv = np.array([1,1])
for depth in range(1,6):
  width = 2 ** depth

  probabilities_0, probabilities_1 = numerical.ComputeDynamicProbability(probabilities_0, probabilities_1, values)
  equiv = numerical.ComputeDynamicEquivalenceClassSizes(equiv)

  hamming_probabilities_0 = numerical.ComputeHammingProbabilities(hamming_probabilities_0, width, values)
  hamming_probabilities_1 = numerical.ComputeHammingProbabilities(hamming_probabilities_1, width, values)

  mu0 = numerical.mean(hamming_probabilities_0)
  mu1 = numerical.mean(hamming_probabilities_1)
  distance = mu1 - mu0
  tv_hamming = TotalVariationHamming(hamming_probabilities_0, hamming_probabilities_1)
  tv = TotalVariation(probabilities_0, probabilities_1, equiv)
  print(width, distance, tv_hamming, tv, tv_hamming - tv)


    
  LowerBound(hamming_probabilities_0, hamming_probabilities_1, distance)

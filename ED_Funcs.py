import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import itertools
import warnings
import math
from numba import njit, prange

pi = np.pi

def Gera_ind(n):
  import itertools

  labels = []
  for label in itertools.product(range(1, 5), repeat=n): #Este loop faz um loop recursivo, que serve para automatizar o processo de escolha do número de sítios
    labels.append(list(label))

  labels = np.array(labels)

  return labels

#@njit
def Sym(labels, n, S, N_e):
  labelsS = np.zeros((len(labels), len(labels[0])))
  labelsN = np.zeros((len(labels), len(labels[0])))
  for i in range(len(labels)):
    for j in range(n):
      if labels[i][j] == 1:
        labelsS[i][j] = 0
        labelsN[i][j] = 0
      elif labels[i][j] == 2:
        labelsS[i][j] = 1
        labelsN[i][j] = 1
      elif labels[i][j] == 3:
        labelsS[i][j] = -1
        labelsN[i][j] = 1
      elif labels[i][j] == 4:
        labelsS[i][j] = 0
        labelsN[i][j] = 2


  NS = np.zeros((len(labels), 2))
  for i in range(len(labels)):
    NS[i] = [sum(labelsN[i]), sum(labelsS[i])]

  NS = np.array(NS)

  states = []
  labels = list(labels)

  for i in range(len(NS)):
    if NS[i][0] == N_e and NS[i][1] == S:
      states.append(list(labels[i]))

  return states

@njit()
def C(sit, spin, state, n):
  new_state = np.zeros(n, dtype = np.int32)

  for i in range(n):
    if i == sit:
      if state[sit] == 2 and spin == 2:
        new_state[sit] = 1
      elif state[sit] == 4 and spin == 2:
        new_state[sit] = 3
      elif state[sit] == 4 and spin == 3:
        new_state[sit] = 2
      elif state[sit] == 3 and spin == 3:
        new_state[sit] = 1
      elif state[sit] == 4 and spin == 4:
        new_state[sit] = 1
      elif state[sit] == 1:
        new_state[sit] = 0
      else:
        pass
    else:
      new_state[i] = state[i]

  return new_state

@njit()
def C_dag(sit, spin, state, n):
  new_state1 = np.zeros(n)

  for i in range(n):
    if i == sit:
      if state[sit] == 1 and spin == 2:
        new_state1[sit] = 2
      if state[sit] == 3 and spin == 2:
        new_state1[sit] = 4
      elif state[sit] == 1 and spin == 3:
        new_state1[sit] = 3
      elif state[sit] == 2 and spin == 3:
        new_state1[sit] = 4
      elif state[sit] == 1 and spin == 4:
        new_state1[sit] = 4
      elif state[sit] == 4:
        new_state1[sit] = 0
      elif state[sit] == 2 and spin == 2:
        new_state1[sit] = 0
      else:
        pass
    else:
      new_state1[i] = state[i]


  return new_state1

@njit(parallel = True)
def H_hop(n, n_states, t, labels):
  H1 = np.zeros((n_states, n_states))

  for k in range(n-1):
    for i in prange(len(labels)):
      C0 = C(k, 2, labels[i], n)
      Cdag = C_dag(k+1, 2, C0, n)

      for j in range(len(labels)):
        count = 0
        for l in range(n):
          if labels[j][l] == Cdag[l]:
            count += 1
          if count == n:
            H1[i][j] += -t

      C0 = C(k, 3, labels[i], n)
      Cdag = C_dag(k+1, 3, C0, n)

      for j in range(len(labels)):
        count = 0
        for l in range(n):
          if labels[j][l] == Cdag[l]:
            count += 1
          if count == n:
            H1[i][j] += -t

  H1 = H1 + np.transpose(H1)
  return H1

@njit()
def H_int(n, n_states, U, labels):
  H2 = np.zeros((n_states, n_states))

  for k in range(n):
    for i in prange(len(labels)):
      C0 = C(k, 2, labels[i], n)
      Cdag = C_dag(k, 2, C0, n)

      C0_ = C(k, 3, Cdag, n)
      Cdag_ = C_dag(k, 3, C0_, n)

      for j in range(len(labels)):
        count = 0
        for l in range(n):
          if labels[j][l] == Cdag_[l]:
            count += 1
            if count == n:
              H2[i][j] += U


  return H2

@njit()
def H_mu(n, n_states, mu, labels):
  H3 = np.zeros((n_states, n_states))

  for k in range(n):
    for i in prange(len(labels)):
      C0 = C(k, 2, labels[i], n)
      Cdag = C_dag(k, 2, C0, n)

      for j in range(len(labels)):
        count = 0
        for l in range(n):
          if labels[j][l] == Cdag[l]:
            count += 1
            if count == n:
              H3[i][j] += -mu
      C0 = C(k, 3, labels[i], n)
      Cdag = C_dag(k, 3, C0, n)

      for j in range(len(labels)):
        count = 0
        for l in range(n):
          if labels[j][l] == Cdag[l]:
            count += 1
            if count == n:
              H3[i][j] += -mu
      
  return H3

@njit
def Gera_Ham(n, labS, mu, t, U):
  Hmu = H_mu(n, len(labS), mu, labS)
  Ht = H_hop(n, len(labS), t, labS)
  HU = H_int(n, len(labS), U, labS)

  H = Hmu + Ht + HU

  return H

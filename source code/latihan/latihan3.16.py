# Modifikasi Example 3.22 - Q_test_8states.py
# Desain routing diagram sendiri dengan 8 state (0-7), goal = 7
import numpy as np
import pylab as plt
from Q_Utils import *

# Setting up Parameters =========================================
# Routing list untuk 8 state:
# 0-1, 0-2, 1-3, 2-3, 2-4, 3-5, 4-5, 4-6, 5-7, 6-7
points_list = [(0,1), (0,2), (1,3), (2,3), (2,4), (3,5), (4,5), (4,6), (5,7), (6,7)]

# Goal ditetapkan pada state 7
goal = 7

# Tampilkan graf routing
showgraph(points_list)

# Jumlah titik pada matriks R (8 state: 0 sampai 7)
MATRIX_SIZE = 8

# Buat matriks R
R = createRmat(MATRIX_SIZE, points_list, goal)

# Buat matriks Q
Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))

# Parameter learning
gamma = 0.8

# Training ========================================================
scores = []
for i in range(700):
    # Pilih current_state (titik awal) secara acak
    current_state = np.random.randint(0, int(Q.shape[0]))
    # Cari semua aksi (langkah) yang tersedia
    available_act = available_actions(R, current_state)
    # Pilih aksi berikutnya secara acak
    action = sample_next_action(available_act)
    # Update matriks Q
    score = update(R, Q, current_state, action, gamma)
    scores.append(score)
    print('Score:', str(score))

print("Trained Q matrix:")
print(Q / np.max(Q) * 100)

# Testing =========================================================
current_state = 0
steps = [current_state]
while current_state != goal:
    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index))
    else:
        next_step_index = int(next_step_index[0])
    steps.append(next_step_index)
    current_state = next_step_index

# Display Results ===================================================
print("Most efficient path:")
print(steps)
plt.figure()
plt.plot(scores)
plt.title("Convergence of Q-learning (8 states)")
plt.xlabel("Iteration")
plt.ylabel("Score")
plt.show()
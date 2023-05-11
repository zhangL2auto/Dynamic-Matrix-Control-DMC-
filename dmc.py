'''
Author: zhangL
Date: 2023-03-23 14:06:46
LastEditors: zhangL
LastEditTime: 2023-05-11 14:10:54
FilePath: /control_model/dmc.py
'''

import itertools
import numpy as np
from scipy.signal import cont2discrete, TransferFunction, step, tf2ss, lti
import matplotlib.pyplot as plt

def dmc():  # sorcery skip: remove-redundant-slice-index

# Parameters
    p = 10
    m = 7
    n = 50
# System
    num = [5]
    den = [8, 1]
    dt = 0.1
    steps = 500

    sys = TransferFunction(num, den)
    sys2 = lti(num, den) 
    AA, B, C, D = tf2ss(num, den)
    ad, bd, cd, dd, dt = cont2discrete((AA, B, C, D), dt, method="zoh")
# Step Response
    t, a = step(sys, N = n)
    # a = np.array([0, 0.815, 1.2422, 1.4553, 1.5614, 1.6143, 1.6406 ,1.6537, 1.6602, 1.6634 ,1.6651, 1.6659, 1.6663 ,1.6665, 1.6666 ,1.6666, 1.6666, 1.6667, 1.6667, 1.6667])
    a = a.reshape(-1,1)

# Dynamic matrix a
    A = np.zeros((p, m))
    for i, j in itertools.product(range(p), range(m)):
        if j <= i:
            A[i, j] = a[i-j]

# Optimal u
    Q = np.eye(p)
    R = 0.01 * np.eye(m)
    c = np.zeros((1,m))
    if len(c[0]) == 1:
        c[0] = 1
    else:
        c[0][0] = 1
        
    d = np.linalg.inv(np.transpose(A).dot(Q).dot(A) + R).dot(np.transpose(A)).dot(Q)

# Error matrix and S
    alpha = 0.75
    H = np.ones((n,1))
    H = alpha * H
    H[0] = 1

    S = np.zeros((p, p))
    for i in range(p-1):
        S[i, i+1] = 1
    S[p-1, p-1] = 1
# Referce and output

    # yr = np.sin(np.linspace(0, 15, steps))
    t, yr = step(sys2, N = steps)
    # yr = np.sin(np.linspace(0, 20 ,steps)).reshape(-1,1)

    yr = yr.reshape(-1,1)
    yp = np.zeros((p, 1))

    yt = np.zeros((steps, 1))
    u_out = np.zeros((steps, 1))

# First control
    xs0 = np.array([0]).reshape(1,1)
    xs1 = ad.dot(xs0)
    yt[0] = cd.dot(xs1)
    xs0 = xs1

    ycor = yp + (H[:p]*(yt[0] - yp[0]))
    yp = np.dot(S, ycor)
    u = d.dot(yr[:p] - yp[:p]) # sum 1-P
    yp = yp + A.dot((u))

    du = c.dot(u)
    u_out[0] = du

# Loop control
    for i in range(1, steps-p):
        xs1 = ad.dot(xs0) + bd.dot(u_out[i-1]).reshape(-1,1)
        yt[i]= cd.dot(xs1) + dd.dot(u_out[i-1])
        xs0 = xs1

        ycor = yp + (H[:p]*(yt[i] - yp[0]))
        yp = S.dot(ycor)

        u = d.dot(yr[i:p+i] - yp[:p]) # sum 1-P
        yp = yp + A.dot((u))
        du = c.dot(u)
    
        u_out[i] = u_out[i-1] + du

# Plot
    plt.figure("DMC")
    plt.plot(yr[:steps-p], 'r', label = "参考轨迹")
    plt.plot(yt[:steps-p], 'g', label = '实际输出')
    plt.plot(u_out[:steps-p], 'y', label = "控制输出")
    plt.legend()
    plt.show()

def pid(kp = 5, ki = 5, kd = 2):

    num = [5]
    den = [8, 1]
    steps = 500
    sys = lti(num, den)
    t, yr = step(sys, N= steps)

    # yr = np.sin(np.linspace(0, 20 ,steps)).reshape(-1,1)

    y_hat = 0
    e = yr[0] - y_hat
    se = e
    de = e
    yt = [y_hat]
    for i in range(1, (steps)):
        y_hat = kp * e *0.1 + ki * se*0.1 + kd * de*0.01
        e2 = yr[i] - y_hat
        se += e2
        de = e2 - e
        e = e2
        yt.append(y_hat)
    # a = yt - y_hat 
    
    # print(yt)
    plt.figure("PID")
    plt.plot(yt, 'g', label = '实际输出')
    plt.plot(yr, 'r', label = "参考轨迹")
    plt.show()

def main():
    dmc()
    # pid()

if __name__ == "__main__":
    main()
    # num = [5]
    # den = [8, 1]
    # steps = 500
    # sys = lti(num, den)
    # t, yr = step(sys, N= steps)
    # plt.plot(t, yr)
    # plt.show()
import numpy as np
import matplotlib.pyplot as plt
import time
# np.random.seed(0)

# 目标函数
def objective_function(x):
    p = a * x[:, None] + b
    D1 = np.exp(alpha * p + beta)
    D2_1 = np.sum(np.exp(alpha.T * p[:,:, None] + beta.T), axis=2)
    D2_2 = gamma * (np.sum(np.exp(np.repeat(alpha[:, None], K, axis=1) * np.repeat(p[:, None], K, axis=1) 
                                  + np.repeat(beta[:, None], K, axis=1)), axis=2) 
                            - np.exp(alpha * p + beta))
    D2 = 1 + D2_1 + D2_2
    Dik = D1/D2
    Rp = np.sum(p * Dik)
    return -Rp

# 梯度差分
def difference_gradient(func, xk):
    grad = []
    for i in range(len(xk)):
        x = xk.copy()
        x[i] = np.random.normal(x[i], 0.01*abs(x[i]))
        epsilon = x[i] - xk[i]
        gradient_estimate = (func(x) - func(xk)) / (x[i] - xk[i])
        grad.append(gradient_estimate)
    return np.array(grad)

# 随机梯度下降算法（加速）
def sgd_with_agd(xk, learning_rate, iterations = 100000, tol = 1e-10):
    yk = xk.copy()
    gradient_norms = []
    distances_to_optimal_value = []
    distances_to_optimal_solution = []
    optimal_value = []
    optimal_solution = []
    for k in range(iterations):
        current_value = objective_function(xk)
        grad = difference_gradient(objective_function, xk)
        ykk = xk - learning_rate * grad
        xk  = ykk + k/(k+3)*(ykk - yk)
        yk  = ykk
        next_value = objective_function(xk)
        # process
        gradient_norms.append(np.linalg.norm(grad))
        distances_to_optimal_value.append(abs(current_value - next_value))
        optimal_value.append(next_value)
        distances_to_optimal_solution.append(np.linalg.norm(learning_rate * grad))
        optimal_solution.append(xk)
        # 判断误差 
        if k % 1000 == 0:
            print('第', k+1, '步:  \n f(x): ', next_value)
        while abs(current_value - next_value) < tol:
            print('第', k+1, '步: \n x: ', xk, '\n f(x): ', next_value)
            return gradient_norms, distances_to_optimal_value, distances_to_optimal_solution, optimal_value, optimal_solution
    print('第', k+1, '步: \n x: ', xk, '\n f(x): ', next_value)
    return gradient_norms, distances_to_optimal_value, distances_to_optimal_solution, optimal_value, optimal_solution

# 自适应矩估计算法
def adam_optimizer(xk, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, iterations = 100000, tol = 1e-10):
    m = np.zeros_like(xk)
    v = np.zeros_like(xk)
    current_value = np.inf
    gradient_norms = []
    distances_to_optimal_value = []
    distances_to_optimal_solution = []
    optimal_value = []
    optimal_solution = []
    for k in range(1, iterations):
        grad = difference_gradient(objective_function, xk)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * np.square(grad)
        m_hat = m / (1 - beta1**k)
        v_hat = v / (1 - beta2**k)
        xk = xk - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        next_value = objective_function(xk)
        # process
        gradient_norms.append(np.linalg.norm(grad))
        distances_to_optimal_value.append(abs(current_value - next_value))
        optimal_value.append(next_value)
        distances_to_optimal_solution.append(np.linalg.norm(learning_rate * grad))
        optimal_solution.append(xk)
        # 判断误差 
        if k % 1000 == 1 :
            print('第', k, '步:  \n f(x): ', next_value)
        while abs(current_value - next_value) < tol:
            print('第', k, '步: \n x: ', xk, '\n f(x): ', next_value)
            return gradient_norms, distances_to_optimal_value, distances_to_optimal_solution, optimal_value, optimal_solution
        current_value = next_value
    print('第', k+1, '步: \n x: ', xk, '\n f(x): ', next_value)
    return gradient_norms, distances_to_optimal_value, distances_to_optimal_solution, optimal_value, optimal_solution


if __name__ == "__main__":
    # 参数配置
    n = 100
    K = 20

    alpha = np.random.uniform(-1.5, -3, size=(n, K))
    beta = np.random.uniform(-1, -2, size=(n, K))
    a = np.random.uniform(0.8, 1, size=(n, K))
    b = np.random.uniform(0, 0.1, size=(n, K))
    gamma = np.random.uniform(0.2, 0.6)

    p0 = np.random.rand(n)
    learning_rate = 0.001

    # SGD
    print("--------------------SGD--------------------")
    start = time.time()
    SGDwithAGD_solution = sgd_with_agd(p0, learning_rate, tol=1e-8)
    print('time:',time.time()-start)

    # Adam
    print("--------------------Adam-------------------")
    start = time.time()
    Adam_solution = adam_optimizer(p0, learning_rate, tol=1e-8)
    print('time:',time.time()-start)
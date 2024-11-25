import numpy as np
import matplotlib.pyplot as plt
import time
# 设置随机种子以确保结果可复现
# np.random.seed(0)

# 目标函数
def objective_function(x):
    x1, x2, x3 = x
    f = x1**4 - 6*x1**3*x2 + 2*x1**3*x3 + 6*x1**2*x3**2 + 9*x1**2*x2**2 - 6*x1**2*x2*x3 - 14*x1*x2*x3**2 + 4*x1*x3**3 + 5*x3**4 - 7*x2**2*x3**2 + 16*x2**4
    return f

# 目标函数的梯度
def gradient_function(x):
    x1, x2, x3 = x
    grad1 = 4*x1**3 - 18*x1**2*x2 + 6*x1**2*x3 + 12*x1*x3**2 + 18*x1*x2**2 - 12*x1*x2*x3 - 14*x2*x3**2 + 4*x3**3 
    grad2 = - 6*x1**3 + 18*x1**2*x2 - 6*x1**2*x3 - 14*x1*x3**2 - 14*x2*x3**2 + 64*x2**3
    grad3 = 2*x1**3 + 12*x1**2*x3 - 6*x1**2*x2 - 28*x1*x2*x3 + 12*x1*x3**2 + 20*x3**3 - 14*x2**2*x3
    return np.array([grad1, grad2, grad3])

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
def sgd_with_agd(xk, learning_rate, iterations = 100000, tol = 1e-15):
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
        optimal_value.append(current_value)
        distances_to_optimal_solution.append(np.linalg.norm(learning_rate * grad))
        optimal_solution.append(xk)
        # 判断误差 
        if k % 10000 == 0:
            print('第', k+1, '步:  \n f(x): ', next_value)
        while abs(current_value - next_value) < tol:
            print('第', k+1, '步: \n x: ', xk, '\n f(x): ', next_value)
            return gradient_norms, distances_to_optimal_value, distances_to_optimal_solution, optimal_value, optimal_solution
    print('第', k+1, '步: \n x: ', xk, '\n f(x): ', next_value)
    return gradient_norms, distances_to_optimal_value, distances_to_optimal_solution, optimal_value, optimal_solution

def adam_optimizer(xk, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, iterations = 100000, tol = 1e-15):
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
        optimal_value.append(current_value)
        distances_to_optimal_solution.append(np.linalg.norm(learning_rate * grad))
        optimal_solution.append(xk)
        # 判断误差 
        if k % 10000 == 0:
            print('第', k+1, '步:  \n f(x): ', next_value)
        while abs(current_value - next_value) < tol:
            print('第', k+1, '步: \n x: ', xk, '\n f(x): ', next_value)
            return gradient_norms, distances_to_optimal_value, distances_to_optimal_solution, optimal_value, optimal_solution
        current_value = next_value
    print('第', k+1, '步: \n x: ', xk, '\n f(x): ', next_value)
    return gradient_norms, distances_to_optimal_value, distances_to_optimal_solution, optimal_value, optimal_solution


if __name__ == "__main__":
    n = 3
    learning_rate = 0.001 
    xk0 = np.random.rand(n)*5
    print(xk0)

    # SGD
    print("--------------------SGD--------------------")
    start = time.time()
    SGDwithAGD_solution = sgd_with_agd(xk0, learning_rate)
    print('time:',time.time()-start)

    # Adam
    print("--------------------Adam-------------------")
    start = time.time()
    Adam_solution = adam_optimizer(xk0, learning_rate)
    print('time:',time.time()-start)

    # 绘制收敛效果
    gradient_norms1, distances_to_optimal_value1, distances_to_optimal_solution1, optimal_value1, optimal_solution1 = SGDwithAGD_solution
    gradient_norms2, distances_to_optimal_value2, distances_to_optimal_solution2, optimal_value2, optimal_solution2 = Adam_solution
    
    fig, axes = plt.subplots(2,2, figsize=(10, 10))

    axes[0,0].loglog(range(1, len(gradient_norms1)+1), gradient_norms1)
    axes[0,0].loglog(range(1, len(gradient_norms2)+1), gradient_norms2)
    axes[0,0].set_xlabel('Iteration')
    axes[0,0].set_ylabel('Gradient Norm')

    axes[0,1].loglog(range(1, len(distances_to_optimal_value1)+1), distances_to_optimal_value1)
    axes[0,1].loglog(range(1, len(distances_to_optimal_value2)+1), distances_to_optimal_value2)
    axes[0,1].set_xlabel('Iteration')
    axes[0,1].set_ylabel('Distance to Optimal Value')

    axes[1,0].loglog(range(1, len(distances_to_optimal_solution1)+1), distances_to_optimal_solution1)
    axes[1,0].loglog(range(1, len(distances_to_optimal_solution2)+1), distances_to_optimal_solution2)
    axes[1,0].set_xlabel('Iteration')
    axes[1,0].set_ylabel('Distance to Optimal Solution')

    axes[1,1].loglog(range(1, len(optimal_value1)+1), optimal_value1)
    axes[1,1].loglog(range(1, len(optimal_value2)+1), optimal_value2)
    axes[1,1].set_xlabel('Iteration')
    axes[1,1].set_ylabel('Optimal Value')

    plt.tight_layout()
    plt.show()
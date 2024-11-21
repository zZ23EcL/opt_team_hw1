import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
import cvxpy as cp
from sklearn.impute import KNNImputer
from PIL import Image

# 图片灰度转换
def load_image_as_matrix(image_path, size=(3840, 2160)):
    img = Image.open(image_path).convert("L")  # 转换为灰度
    img = img.resize(size)  # 调整尺寸
    return np.array(img)

# 随机遮蔽90%数据
def mask_matrix(matrix, retain_ratio=0.1):
    mask = np.random.rand(*matrix.shape) < retain_ratio
    masked_matrix = np.where(mask, matrix, np.nan)
    return masked_matrix, mask

# SVD矩阵补全
def svd_matrix_completion(masked_matrix, n_components=50):
    # 将 NaN 替换为列均值
    imputer = SimpleImputer(strategy="mean")
    imputed_matrix = imputer.fit_transform(masked_matrix)

    # SVD 分解和恢复
    svd = TruncatedSVD(n_components=n_components)
    low_rank = svd.fit_transform(imputed_matrix)
    recovered_matrix = svd.inverse_transform(low_rank)
    return recovered_matrix
# SVD手搓版本
def svd_manual(A):
    """
    手动实现矩阵的 SVD 分解。
    输入：
        A: 待分解的矩阵 (m x n)
    输出：
        U: 左奇异矩阵 (m x m)
        S: 奇异值 (对角阵形式, m x n)
        V: 右奇异矩阵 (n x n)
    """
    ATA = np.dot(A.T, A)
    AAT = np.dot(A, A.T)

    eigvals_U, U = np.linalg.eigh(AAT)
    U = U[:, ::-1]  # 从大到小排列特征向量
    eigvals_U = eigvals_U[::-1]

    eigvals_V, V = np.linalg.eigh(ATA)
    V = V[:, ::-1]  # 从大到小排列特征向量
    eigvals_V = eigvals_V[::-1]

    singular_values = np.sqrt(np.maximum(eigvals_U, 0))

    S = np.zeros_like(A, dtype=float)
    np.fill_diagonal(S, singular_values)

    return U, S, V.T

def svd_reconstruct(U, S, VT, k):
    """
    使用前 k 个奇异值重构矩阵。
    输入：
        U, S, VT: SVD 分解结果
        k: 使用的奇异值个数
    输出：
        A_k: 重构的矩阵
    """
    return U[:, :k] @ S[:k, :k] @ VT[:k, :]

# 随机抹去图像数据
def mask_image(image, mask_ratio=0.9):
    """
    随机抹去图像数据（用 NaN 表示）。
    输入：
        image: 原始灰度图像矩阵
        mask_ratio: 抹去的像素比例
    输出：
        masked_image: 带有缺失值的图像矩阵
        mask: 掩码矩阵
    """
    mask = np.random.choice([0, 1], size=image.shape, p=[mask_ratio, 1 - mask_ratio])
    masked_image = np.where(mask, image, np.nan)
    return masked_image, mask

def fill_missing_values(image, mask, k=10):
    """
    使用 SVD 对图像缺失值进行补全。
    输入：
        image: 含有 NaN 的图像矩阵
        mask: 掩码矩阵，1 表示观测值，0 表示缺失值
        k: 使用前 k 个奇异值补全
    输出：
        completed_image: 补全后的图像
    """
    # 将缺失值填充为零
    filled_image = np.nan_to_num(image)

    # SVD 分解
    U, S, VT = svd_manual(filled_image)

    # 重构矩阵
    reconstructed_image = svd_reconstruct(U, S, VT, k)

    # 用重构值更新缺失部分
    completed_image = mask * image + (1 - mask) * reconstructed_image
    return completed_image

# 交替最小二乘法 ALS
def als_matrix_completion(masked_matrix, n_components=50, max_iter=200):
    # 替换 NaN 为零
    masked_matrix = np.nan_to_num(masked_matrix)

    # 使用非负矩阵分解 (NMF) 模拟交替最小二乘
    model = NMF(n_components=n_components, max_iter=max_iter, random_state=42)
    W = model.fit_transform(masked_matrix)  # 左矩阵
    H = model.components_  # 右矩阵
    return np.dot(W, H)  # 恢复矩阵

# 核范数最小化 NNM
def nuclear_norm_completion(masked_matrix, mask, lambd=1.0):
    # 定义变量
    X = cp.Variable(masked_matrix.shape)  # 待补全矩阵
    observed = mask * masked_matrix  # 已观测数据

    # 定义目标函数：核范数最小化 + 数据一致性
    objective = cp.Minimize(cp.norm(X, "nuc") + lambd * cp.sum_squares(cp.multiply(mask, X - observed)))
    problem = cp.Problem(objective)
    problem.solve(solver=cp.SCS)  # 选择解算器
    return X.value
# KNN 数据插值
def knn_imputation(masked_matrix, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    return imputer.fit_transform(masked_matrix)

# 可视化
def visualize_results(original, masked, recovered):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Original", "Masked", "Recovered"]
    images = [original, masked, recovered]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# 主函数
image_path = "4kpic1.jpg"  # 替换为图片路径
original_matrix = load_image_as_matrix(image_path)
masked_matrix, mask = mask_matrix(original_matrix, retain_ratio=0.1)
recovered_matrix = svd_matrix_completion(masked_matrix)

# 可视化
visualize_results(original_matrix, masked_matrix, recovered_matrix)


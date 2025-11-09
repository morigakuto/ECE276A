
import numpy as np
from utils import read_canonical_model, load_pc, visualize_icp_result


def find_closest_points(source_points, target_points):
    """
    各ソース点に対して最も近いターゲット点を見つけます。

    Args:
        source_points (np.array): ソース点群 (N, 3)
        target_points (np.array): ターゲット点群 (M, 3)

    Returns:
        tuple: (matched_source, matched_target) - 対応点のペア
    """
    matched_source = []
    matched_target = []

    for source_point in source_points:
        # 各ソース点に対して全ターゲット点との距離を計算
        distances = np.linalg.norm(target_points - source_point, axis=1)
        min_idx = np.argmin(distances)

        matched_source.append(source_point)
        matched_target.append(target_points[min_idx])

    return np.array(matched_source), np.array(matched_target)


def compute_transformation(source_points, target_points):
    """
    Kabschアルゴリズムを使用して最適な回転行列と並進ベクトルを計算します。

    Args:
        source_points (np.array): ソース点群 (N, 3)
        target_points (np.array): ターゲット点群 (N, 3)

    Returns:
        tuple: (R, t) - 回転行列 (3, 3) と並進ベクトル (3,)
    """
    # Step 1: 重心を計算
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)

    # Step 2: 中心化
    centered_source = source_points - source_centroid
    centered_target = target_points - target_centroid

    # Step 3: 共分散行列を計算
    H = centered_source.T @ centered_target

    # Step 4: SVDを使用して回転行列を計算
    U, S, Vt = np.linalg.svd(H)

    # Step 5: リフレクション（鏡像）を修正
    d = np.linalg.det(U @ Vt)
    if d < 0:
        Vt[-1, :] *= -1

    R = Vt.T @ U.T

    # Step 6: 並進ベクトルを計算
    t = target_centroid - R @ source_centroid

    return R, t


def icp(source_pc, target_pc, max_iterations=50, tolerance=1e-5):
    """
    Iterative Closest Point (ICP) アルゴリズムの実装

    Args:
        source_pc (np.array): ソース点群（モデル） (N, 3)
        target_pc (np.array): ターゲット点群（観測データ） (M, 3)
        max_iterations (int): 最大反復回数
        tolerance (float): 収束判定の閾値

    Returns:
        np.array: 4x4の変換行列 (SE(3))
    """
    # 初期変換行列
    pose = np.eye(4)

    # ソース点群のコピー（変換を適用していく）
    transformed_source = source_pc.copy()

    prev_error = float('inf')

    for iteration in range(max_iterations):
        # Step 1: 最近傍点の探索
        matched_source, matched_target = find_closest_points(transformed_source, target_pc)

        # Step 2: 変換行列の計算（Kabschアルゴリズム）
        R, t = compute_transformation(matched_source, matched_target)

        # Step 3: ソース点群を変換
        transformed_source = (R @ transformed_source.T).T + t

        # Step 4: 累積変換行列を更新
        delta_pose = np.eye(4)
        delta_pose[:3, :3] = R
        delta_pose[:3, 3] = t
        pose = delta_pose @ pose

        # Step 5: 収束判定
        # 対応点間の平均距離を計算
        error = np.mean(np.linalg.norm(matched_target - matched_source, axis=1))

        # 収束判定
        if abs(prev_error - error) < tolerance:
            print(f"  ICP converged at iteration {iteration + 1}, error: {error:.6f}")
            break

        prev_error = error

        # 進捗表示（10反復ごと）
        if (iteration + 1) % 10 == 0:
            print(f"  Iteration {iteration + 1}: Mean error = {error:.6f}")

    if iteration == max_iterations - 1:
        print(f"  ICP reached maximum iterations ({max_iterations})")

    return pose


def downsample_point_cloud(point_cloud, sample_ratio=0.3):
    """
    点群をダウンサンプリングします（高速化のため）

    Args:
        point_cloud (np.array): 入力点群 (N, 3)
        sample_ratio (float): サンプリング比率 (0.0 ~ 1.0)

    Returns:
        np.array: ダウンサンプリングされた点群
    """
    n_points = len(point_cloud)
    n_samples = int(n_points * sample_ratio)

    if n_samples < 100:  # 最小点数を確保
        n_samples = min(100, n_points)

    indices = np.random.choice(n_points, n_samples, replace=False)
    return point_cloud[indices]


if __name__ == "__main__":
    # パラメータ設定
    obj_names = ['drill', 'liq_container']  # 両方のオブジェクトでテスト
    num_pc = 4  # 各オブジェクトの点群数

    print("=" * 60)
    print("ECE276A PR2: ICP Warm-up Test")
    print("=" * 60)

    for obj_name in obj_names:
        print(f"\n### Testing with object: {obj_name} ###")

        # モデル（正準形）を読み込み
        source_pc = read_canonical_model(obj_name)
        print(f"Loaded model with {source_pc.shape[0]} points")

        # 各観測データに対してICPを実行
        for i in range(num_pc):
            print(f"\nProcessing point cloud {i}...")

            # 観測データを読み込み
            target_pc = load_pc(obj_name, i)
            print(f"  Loaded target with {target_pc.shape[0]} points")

            # 点群をダウンサンプリング（高速化のため）
            source_downsampled = downsample_point_cloud(source_pc, sample_ratio=0.3)
            target_downsampled = downsample_point_cloud(target_pc, sample_ratio=0.3)
            print(f"  Downsampled: source={len(source_downsampled)}, target={len(target_downsampled)}")

            # ICPで姿勢を推定
            print("  Running ICP...")
            pose = icp(source_downsampled, target_downsampled, max_iterations=50)

            # 結果を表示
            print(f"\nEstimated pose for {obj_name} - point cloud {i}:")
            print("Rotation matrix:")
            print(pose[:3, :3])
            print("Translation vector:")
            print(pose[:3, 3])

            # 推定された回転角度を計算（オイラー角として）
            rotation_matrix = pose[:3, :3]
            # Y-X-Z オイラー角を計算（簡易版）
            theta_x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            theta_y = np.arctan2(-rotation_matrix[2, 0],
                               np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
            theta_z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

            print(f"Rotation angles (degrees): X={np.degrees(theta_x):.2f}, "
                  f"Y={np.degrees(theta_y):.2f}, Z={np.degrees(theta_z):.2f}")

            # 可視化（Open3Dを使用）
            try:
                print("\nVisualizing result...")
                print("(Close the visualization window to continue)")
                visualize_icp_result(source_pc, target_pc, pose)
            except Exception as e:
                print(f"Warning: Visualization failed ({e})")
                print("Continuing without visualization...")

        print(f"\n{'=' * 60}")

    print("\nICP warm-up test completed!")
    print("=" * 60)


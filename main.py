# main.py
import numpy as np
import matplotlib.pyplot as plt

def main():
    # データセット20を使う
    dataset = 20
    
    # === ステップ1: データを読み込む ===
    print("データを読み込んでいます...")
    
    # エンコーダーデータ
    with np.load(f"data/Encoders{dataset}.npz") as data:
        encoder_counts = data["counts"]  # 4 x n の配列
        encoder_stamps = data["time_stamps"]  # n個のタイムスタンプ
    
    # IMUデータ
    with np.load(f"data/Imu{dataset}.npz") as data:
        imu_angular_velocity = data["angular_velocity"]  # 3 x m の配列
        imu_stamps = data["time_stamps"]  # m個のタイムスタンプ
    
    # データの形を確認
    print(f"エンコーダー: {encoder_counts.shape} (4輪 x {encoder_counts.shape[1]}測定)")
    print(f"エンコーダータイムスタンプ: {encoder_stamps.shape}")
    print(f"IMU角速度: {imu_angular_velocity.shape} (3軸 x {imu_angular_velocity.shape[1]}測定)")
    print(f"IMUタイムスタンプ: {imu_stamps.shape}")
    
    # 最初の5個のデータを見てみる
    print("\n最初の5個のエンコーダーカウント:")
    print(encoder_counts[:, :5])
    print("\n最初の5個のエンコーダータイムスタンプ:")
    print(encoder_stamps[:5])
    
    print("\n最初の5個のIMU角速度:")
    print(imu_angular_velocity[:, :5])
        
    # === データをもっと詳しく見る ===
    
    # ロボットが動き始めるのはいつ？
    print("\n=== ロボットはいつ動き始める？ ===")
    for i in range(20):  # 最初の20ステップをチェック
        total_count = np.sum(np.abs(encoder_counts[:, i]))
        if total_count > 0:
            print(f"ステップ {i}: エンコーダーカウント合計 = {total_count}")
            print(f"  各輪: FR={encoder_counts[0,i]}, FL={encoder_counts[1,i]}, "
                  f"RR={encoder_counts[2,i]}, RL={encoder_counts[3,i]}")
            if i >= 10:  # 最初の10個の動きを表示したら終了
                break
    
    # タイムスタンプの時間差を確認
    print("\n=== タイムスタンプの間隔 ===")
    encoder_dt = encoder_stamps[1:6] - encoder_stamps[0:5]
    print(f"エンコーダーの時間間隔（秒）: {encoder_dt}")
    print(f"エンコーダーの周波数: 約 {1/np.mean(encoder_dt):.1f} Hz")
    
    imu_dt = imu_stamps[1:6] - imu_stamps[0:5]
    print(f"\nIMUの時間間隔（秒）: {imu_dt}")
    print(f"IMUの周波数: 約 {1/np.mean(imu_dt):.1f} Hz")
    
    # タイムスタンプの範囲を確認
    print("\n=== データの時間範囲 ===")
    print(f"エンコーダー: {encoder_stamps[0]:.2f} 秒 ～ {encoder_stamps[-1]:.2f} 秒")
    print(f"合計時間: {encoder_stamps[-1] - encoder_stamps[0]:.2f} 秒")
    print(f"IMU: {imu_stamps[0]:.2f} 秒 ～ {imu_stamps[-1]:.2f} 秒")
    print(f"合計時間: {imu_stamps[-1] - imu_stamps[0]:.2f} 秒")
    
    print("\n" + "="*50)
    print("=== オドメトリ計算を開始 ===")
    print("="*50)
    
    # === ステップ1: エンコーダーから線速度を計算 ===
    
    # 定数
    WHEEL_DIAMETER = 0.254  # メートル
    TICKS_PER_REVOLUTION = 360
    METERS_PER_TICK = 0.0022  # 仕様書より
    
    # 姿勢を記録する配列を準備
    num_steps = encoder_counts.shape[1]
    trajectory_x = np.zeros(num_steps)  # x座標
    trajectory_y = np.zeros(num_steps)  # y座標
    trajectory_theta = np.zeros(num_steps)  # 角度（ヨー）
    
    # 初期姿勢
    x, y, theta = 0.0, 0.0, 0.0
    trajectory_x[0] = x
    trajectory_y[0] = y
    trajectory_theta[0] = theta
    
    print(f"初期姿勢: x={x}, y={y}, theta={theta}")
    print(f"処理するステップ数: {num_steps}")
    
    # 各タイムステップをループ
    for i in range(num_steps - 1):  # 最後から2番目まで
        # 現在の時刻
        t_curr = encoder_stamps[i]
        t_next = encoder_stamps[i + 1]
        dt = t_next - t_curr  # 時間差

        # エンコーダーカウントを取得（次のステップの値）
        FR = encoder_counts[0, i+1]  # 前右
        FL = encoder_counts[1, i+1]  # 前左
        RR = encoder_counts[2, i+1]  # 後右 
        RL = encoder_counts[3, i+1]  # 後左

        # 左右の車輪の移動距離を計算（仕様書の公式）
        right_distance = (FR + RR) / 2.0 * METERS_PER_TICK
        left_distance = (FL + RL) / 2.0 * METERS_PER_TICK

        # ロボットの線速度（左右の平均）
        distance = (right_distance + left_distance) / 2.0
        v = distance / dt if dt > 0 else 0.0

        # === IMUから角速度（ヨーレート）を取得 ===
        # t_nextに最も近いIMUのタイムスタンプを見つける
        imu_idx = np.argmin(np.abs(imu_stamps - t_next))

        # z軸（ヨー）の角速度を取得（IMUの3行目）
        omega = imu_angular_velocity[2, imu_idx]  # rad/sec

        # 微分駆動モデルで姿勢を更新
        x = x + v * np.cos(theta) * dt
        y = y + v * np.sin(theta) * dt
        theta = theta + omega * dt

        # 軌跡に記録
        trajectory_x[i+1] = x
        trajectory_y[i+1] = y
        trajectory_theta[i+1] = theta

        # 最初の数ステップをデバッグ出力
        if i < 5 or (i < 100 and distance > 0):
            print(f"Step {i}: dt={dt:.4f}s, v={v:.4f}m/s, omega={omega:.4f}rad/s, "
                  f"distance={distance:.6f}m, x={x:.3f}, y={y:.3f}, theta={theta:.3f}")
    
    print(f"\n最終姿勢: x={x:.3f}, y={y:.3f}, theta={theta:.3f}")
    
    # === 軌跡をプロット ===
    plt.figure(figsize=(10, 8))
    plt.plot(trajectory_x, trajectory_y, 'b-', linewidth=1, label='Robot trajectory')
    plt.plot(trajectory_x[0], trajectory_y[0], 'go', markersize=10, label='Start')
    plt.plot(trajectory_x[-1], trajectory_y[-1], 'ro', markersize=10, label='End')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Robot Odometry (Encoder + IMU)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('trajectory_encoder_imu.png')
    print("\n軌跡を 'trajectory_encoder_imu.png' に保存しました")
    plt.show()

if __name__ == "__main__":
    main()
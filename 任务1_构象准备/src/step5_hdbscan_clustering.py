# -*- coding: utf-8 -*-
"""
ASPA 构象-口袋特征 HDBSCAN 聚类
============================================================
步骤五: HDBSCAN 口袋特征聚类 + UMAP 降维可视化

输入: data/step5/pocket_features.csv (45 行 × 6 列)
  - 构象编号 (conf_01 ~ conf_15)
  - 口袋ID (Pocket1/Pocket2/Pocket3)
  - 可药性评分
  - 口袋体积
  - 口袋深度
  - 疏水表面积

输出: results/step5/ 下的 6 个文件
  1. preprocessed_features.csv    (标准化后特征)
  2. hdbscan_result.csv           (聚类结果, 原始数据+cluster列)
  3. hdbscan_umap.png             (UMAP 散点图)
  4. representative_pockets.csv   (每簇代表性口袋)
  5. top10_pockets.csv            (Top10 可药性口袋)
  6. conformation_summary.csv     (按构象汇总)

运行方式:
  python src/step5_hdbscan_clustering.py

依赖:
  pip install -r requirements.txt
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 列定义
# ============================================================
# 前2列是标识符, 不参与聚类
ID_COLS = ['构象编号', '口袋ID']
# 后4列是聚类特征
FEATURE_COLS = ['可药性评分', '口袋体积', '口袋深度', '疏水表面积']


def run_clustering(input_csv, output_dir):
    """读取口袋特征 -> 质检/标准化 -> HDBSCAN 网格搜索 -> UMAP/统计输出"""
    os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # 第1步: 读取数据
    # ============================================================
    print("=" * 60)
    print("第1步: 读取口袋特征数据")
    print("=" * 60)

    # 文件可能是 GBK 编码, 默认 UTF-8, 需要尝试多种编码
    df = None
    used_enc = None
    for enc in ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']:
        try:
            df = pd.read_csv(input_csv, encoding=enc)
            used_enc = enc
            break
        except Exception:
            continue

    if df is None:
        raise IOError("无法读取输入文件: %s" % input_csv)
    print(f"✅ 读取成功 (编码: {used_enc})")

    print(f"数据规模: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"构象数: {df['构象编号'].nunique()}")
    print(f"每构象口袋数: {df.groupby('构象编号').size().iloc[0]}")
    print(f"特征列: {FEATURE_COLS}")

    print("\n前10行:")
    print(df.head(10).to_string())

    print("\n特征统计:")
    print(df[FEATURE_COLS].describe().round(2).to_string())

    # ============================================================
    # 第2步: 数据质量检查
    # ============================================================
    print("\n" + "=" * 60)
    print("第2步: 数据质量检查")
    print("=" * 60)

    missing = df[FEATURE_COLS].isnull().sum()
    if missing.sum() == 0:
        print("✅ 无缺失值")
    else:
        print(f"⚠️ 缺失值:\n{missing[missing > 0]}")
        df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
        print("已用中位数填补")

    # ============================================================
    # 第3步: 标准化 (Z-score)
    # ============================================================
    print("\n" + "=" * 60)
    print("第3步: 标准化 (StandardScaler)")
    print("=" * 60)

    X_raw = df[FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 保存标准化后的数据
    df_scaled = df[ID_COLS].copy()
    df_scaled[FEATURE_COLS] = X_scaled
    df_scaled.to_csv(os.path.join(output_dir, 'preprocessed_features.csv'), index=False)
    print("✅ 标准化后数据已保存: preprocessed_features.csv")

    print("\n标准化后统计 (均值≈0, 标准差≈1):")
    print(df_scaled[FEATURE_COLS].describe().round(3).to_string())

    # ============================================================
    # 第4步: HDBSCAN聚类 (参数网格搜索)
    # ============================================================
    print("\n" + "=" * 60)
    print("第4步: HDBSCAN聚类")
    print("=" * 60)

    try:
        import hdbscan
        print("使用 hdbscan 库")
        use_sklearn = False
    except ImportError:
        from sklearn.cluster import HDBSCAN
        print("hdbscan库未安装, 使用sklearn的HDBSCAN")
        use_sklearn = True

    print(f"\n样本数: {len(X_scaled)} (45个构象-口袋对)")
    print("参数网格搜索 (轮廓系数选最优参数):")

    best_score = -1
    best_params = None
    best_labels = None
    best_n_clusters = None
    best_n_noise = None

    # 45个样本, min_cluster_size 设 2-10
    param_grid = [
        {'min_cluster_size': mcs, 'min_samples': ms}
        for mcs in [2, 3, 4, 5, 6, 8, 10]
        for ms in [None, 2, 3, 4, 5]
    ]

    for params in param_grid:
        try:
            if not use_sklearn:
                clusterer = hdbscan.HDBSCAN(**params, metric='euclidean')
            else:
                kwargs = {'min_cluster_size': params['min_cluster_size']}
                if params['min_samples'] is not None:
                    kwargs['min_samples'] = params['min_samples']
                clusterer = HDBSCAN(**kwargs)

            labels = clusterer.fit_predict(X_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            if n_clusters >= 2 and n_noise < len(X_scaled):
                score = silhouette_score(X_scaled, labels)
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_labels = labels
                    best_n_clusters = n_clusters
                    best_n_noise = n_noise
        except Exception:
            continue

    print(f"\n✅ 最优参数: {best_params}")
    print(f"   簇数: {best_n_clusters}")
    print(f"   噪声点: {best_n_noise}")
    print(f"   轮廓系数: {best_score:.4f}")

    if best_score > 0.5:
        print("   评估: 聚类效果好 (>0.5)")
    elif best_score > 0.3:
        print("   评估: 聚类效果可接受 (>0.3)")
    else:
        print("   评估: 聚类效果一般 (<0.3)")

    # 保存聚类结果
    df_result = df.copy()
    df_result['cluster'] = best_labels
    df_result.to_csv(os.path.join(output_dir, 'hdbscan_result.csv'), index=False)
    print("\n✅ 聚类结果已保存: hdbscan_result.csv")

    # ============================================================
    # 第5步: 各簇详细统计
    # ============================================================
    print("\n" + "=" * 60)
    print("第5步: 各簇详细统计")
    print("=" * 60)

    for c in sorted(set(best_labels)):
        mask = best_labels == c
        count = mask.sum()
        cluster_data = df_result.loc[mask]

        if c == -1:
            print(f"\n  噪声点 (-1): {count} 个")
            print(f"    构象: {list(cluster_data['构象编号'])}")
        else:
            print(f"\n  簇 {c}: {count} 个口袋")
            print(f"    平均可药性评分: {cluster_data['可药性评分'].mean():.2f}")
            print(f"    平均口袋体积: {cluster_data['口袋体积'].mean():.1f}")
            print(f"    平均口袋深度: {cluster_data['口袋深度'].mean():.1f}")
            print(f"    平均疏水表面积: {cluster_data['疏水表面积'].mean():.1f}")
            print(f"    构象列表: {list(cluster_data['构象编号'])}")
            print(f"    口袋列表: {list(cluster_data['口袋ID'])}")

    # ============================================================
    # 第6步: UMAP降维可视化
    # ============================================================
    print("\n" + "=" * 60)
    print("第6步: UMAP降维可视化")
    print("=" * 60)

    import umap
    reducer = umap.UMAP(n_components=2, random_state=42,
                        n_neighbors=min(15, len(X_scaled) - 1))
    X_2d = reducer.fit_transform(X_scaled)

    # 画UMAP图
    fig, ax = plt.subplots(figsize=(14, 10))
    colors = plt.cm.Set1(np.linspace(0, 1, max(best_n_clusters, 1) + 1))

    for c in sorted(set(best_labels)):
        mask = best_labels == c
        if c == -1:
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c='gray',
                       label=f'Noise ({mask.sum()})', alpha=0.3, s=80, marker='x')
        else:
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors[c]],
                       label=f'Cluster {c} ({mask.sum()})', alpha=0.7, s=120)

    # 标注每个点: 构象编号+口袋ID
    for i in range(len(df)):
        label = f"{df.iloc[i]['构象编号']}\n{df.iloc[i]['口袋ID']}"
        ax.annotate(label, (X_2d[i, 0], X_2d[i, 1]),
                    fontsize=6, ha='center', va='bottom')

    ax.set_title(f'HDBSCAN Clustering of ASPA Pocket Features\n'
                 f'(min_cluster_size={best_params["min_cluster_size"]}, '
                 f'min_samples={best_params["min_samples"]}, '
                 f'silhouette={best_score:.3f})', fontsize=12)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hdbscan_umap.png'), dpi=150)
    plt.close()
    print("✅ UMAP图已保存: hdbscan_umap.png")

    # ============================================================
    # 第7步: 提取代表性口袋
    # ============================================================
    print("\n" + "=" * 60)
    print("第7步: 提取代表性口袋 (每簇离中心最近)")
    print("=" * 60)

    representative = []
    for c in sorted(set(best_labels)):
        if c == -1:
            continue
        mask = best_labels == c
        cluster_df = df_result.loc[mask].copy()
        center = cluster_df[FEATURE_COLS].mean()
        cluster_df['dist_to_center'] = np.sqrt(
            ((cluster_df[FEATURE_COLS] - center) ** 2).sum(axis=1)
        )
        rep = cluster_df.loc[cluster_df['dist_to_center'].idxmin()]
        representative.append({
            'cluster': c,
            '构象编号': rep['构象编号'],
            '口袋ID': rep['口袋ID'],
            '可药性评分': rep['可药性评分'],
            '口袋体积': rep['口袋体积'],
            '口袋深度': rep['口袋深度'],
            '疏水表面积': rep['疏水表面积'],
            'dist_to_center': rep['dist_to_center'],
            'cluster_size': int(mask.sum())
        })

    rep_df = pd.DataFrame(representative)
    rep_df.to_csv(os.path.join(output_dir, 'representative_pockets.csv'), index=False)
    print("✅ 代表性口袋已保存: representative_pockets.csv")
    print(rep_df.to_string(index=False))

    # ============================================================
    # 第8步: 按可药性评分排序Top10
    # ============================================================
    print("\n" + "=" * 60)
    print("第8步: Top 10 口袋 (按可药性评分排序)")
    print("=" * 60)

    top10 = df_result.nlargest(10, '可药性评分')[ID_COLS + FEATURE_COLS + ['cluster']]
    print(top10.to_string(index=False))
    top10.to_csv(os.path.join(output_dir, 'top10_pockets.csv'), index=False)

    # ============================================================
    # 第9步: 按构象汇总
    # ============================================================
    print("\n" + "=" * 60)
    print("第9步: 按构象汇总")
    print("=" * 60)

    conf_summary = df_result.groupby('构象编号').agg({
        '可药性评分': ['mean', 'max'],
        '口袋体积': ['mean', 'max'],
        '口袋深度': 'mean',
        '疏水表面积': 'mean',
        'cluster': lambda x: list(x)
    }).round(2)
    conf_summary.columns = ['可药性均值', '可药性最大', '体积均值', '体积最大',
                            '深度均值', '疏水均值', '簇分布']
    print(conf_summary.to_string())
    conf_summary.to_csv(os.path.join(output_dir, 'conformation_summary.csv'))

    print("\n" + "=" * 60)
    print("全部完成! 输出文件在 results/step5/ 文件夹下:")
    print("  1. preprocessed_features.csv (标准化特征)")
    print("  2. hdbscan_result.csv (聚类结果)")
    print("  3. hdbscan_umap.png (UMAP散点图)")
    print("  4. representative_pockets.csv (每簇代表性口袋)")
    print("  5. top10_pockets.csv (Top10口袋)")
    print("  6. conformation_summary.csv (构象汇总)")
    print("=" * 60)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    input_csv = os.path.join(root, 'data', 'step5', 'pocket_features.csv')
    output_dir = os.path.join(root, 'results', 'step5')
    run_clustering(input_csv, output_dir)


if __name__ == '__main__':
    main()

"""
测试对比学习风格 IK 的泛化性

测试项目:
1. 风格编码器对齐程度 (pred_style vs true_style 的相似度)
2. 只用 poses 的 IK 误差 vs 用 joints 的 IK 误差
3. 跨人泛化性测试
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.insert(0, '/home/bonuli/Pieper')
sys.path.insert(0, '/home/bonuli/Pieper/2003')
sys.path.insert(0, '/home/bonuli/Pieper/2103')  # 必须最后插入，确保在最前面

from model import ContrastiveStyleIK, NormalizationLayer
from dataset_generalized import create_windowed_dataloaders


def compute_style_similarity(pred_style, true_style):
    """计算风格余弦相似度"""
    pred_norm = F.normalize(pred_style, dim=1)
    true_norm = F.normalize(true_style, dim=1)
    return (pred_norm * true_norm).sum(dim=1).mean().item()


def test_style_alignment(model, dataloader, norm_layer, device):
    """
    测试风格编码器的对齐程度
    
    关键指标:
    - 只用 poses 推断的风格 vs 用 joints 提取的风格 的相似度
    - 相似度越高，说明模型成功学会了从末端轨迹推断运动风格
    """
    model.eval()
    
    all_pred_styles = []
    all_true_styles = []
    all_similarities = []
    
    with torch.no_grad():
        for batch_X, _, _ in dataloader:
            batch_X = batch_X.to(device)
            
            history_poses = batch_X[:, :, :7]
            history_joints = batch_X[:, :, 7:]
            
            # 归一化
            history_poses_norm = norm_layer.normalize_history_poses(history_poses)
            history_joints_norm = norm_layer.normalize_joints(history_joints)
            
            # 两种风格
            pred_style = model.extract_pose_style(history_poses_norm)
            true_style = model.extract_joint_style(history_joints_norm)
            
            # 计算相似度
            sim = compute_style_similarity(pred_style, true_style)
            all_similarities.append(sim)
            
            all_pred_styles.append(pred_style.cpu())
            all_true_styles.append(true_style.cpu())
    
    avg_similarity = np.mean(all_similarities)
    
    print("\n" + "=" * 60)
    print("风格编码器对齐测试")
    print("=" * 60)
    print(f"平均余弦相似度: {avg_similarity:.4f}")
    print(f"解释: ")
    print(f"  > 0.9: 优秀 - 模型成功学会从末端轨迹推断风格")
    print(f"  0.8-0.9: 良好 - 大部分风格信息被捕捉")
    print(f"  0.6-0.8: 一般 - 部分风格信息丢失")
    print(f"  < 0.6: 较差 - 风格编码器需要更多训练")
    
    # 返回所有风格向量用于可视化
    all_pred_styles = torch.cat(all_pred_styles, dim=0).numpy()
    all_true_styles = torch.cat(all_true_styles, dim=0).numpy()
    
    return avg_similarity, all_pred_styles, all_true_styles


def test_ik_error_comparison(model, dataloader, norm_layer, device):
    """
    比较两种模式的 IK 误差:
    1. 只用 poses (实际使用方式)
    2. 用 joints + poses (理想上界)
    """
    model.eval()
    
    errors_pose_only = []
    errors_with_joints = []
    
    with torch.no_grad():
        for batch_X, batch_y, _ in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            history_poses = batch_X[:, :, :7]
            history_joints = batch_X[:, :, 7:]
            target_pose = batch_y[:, :7]
            target_angles = batch_y[:, 7:]
            
            # 归一化
            target_pose_norm = norm_layer.normalize_pose(target_pose)
            history_poses_norm = norm_layer.normalize_history_poses(history_poses)
            history_joints_norm = norm_layer.normalize_joints(history_joints)
            
            # 模式 1: 只用 poses
            pred_1, _ = model(target_pose_norm, history_poses_norm, 
                            history_joints=None, mode='inference', return_aux=True)
            pred_1_denorm = norm_layer.denormalize_joint(pred_1)
            error_1 = torch.mean((pred_1_denorm - target_angles) ** 2, dim=1)
            errors_pose_only.extend(error_1.cpu().numpy())
            
            # 模式 2: 用 joints
            pred_2, _ = model(target_pose_norm, history_poses_norm,
                            history_joints_norm, mode='training', return_aux=True)
            pred_2_denorm = norm_layer.denormalize_joint(pred_2)
            error_2 = torch.mean((pred_2_denorm - target_angles) ** 2, dim=1)
            errors_with_joints.extend(error_2.cpu().numpy())
    
    mean_error_1 = np.mean(errors_pose_only)
    mean_error_2 = np.mean(errors_with_joints)
    gap = (mean_error_1 - mean_error_2) / mean_error_2 * 100
    
    print("\n" + "=" * 60)
    print("IK 误差对比")
    print("=" * 60)
    print(f"只用 poses (实际使用):  {mean_error_1:.6f}")
    print(f"用 joints (理想上界):   {mean_error_2:.6f}")
    print(f"差距: {gap:+.1f}%")
    print(f"解释:")
    print(f"  差距 < 5%:  优秀 - 泛化性极好")
    print(f"  差距 5-15%: 良好 - 可接受的泛化性")
    print(f"  差距 15-30%: 一般 - 有一定信息损失")
    print(f"  差距 > 30%:  较差 - 需要改进")
    
    return mean_error_1, mean_error_2


def visualize_styles(pred_styles, true_styles, save_path='style_visualization.png'):
    """
    可视化风格向量（使用 t-SNE）
    """
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        
        # 合并用于 t-SNE
        all_styles = np.vstack([pred_styles[:1000], true_styles[:1000]])
        
        # t-SNE 降维
        tsne = TSNE(n_components=2, random_state=42)
        embeddings = tsne.fit_transform(all_styles)
        
        # 分离
        n = len(pred_styles[:1000])
        pred_emb = embeddings[:n]
        true_emb = embeddings[n:]
        
        # 画图
        plt.figure(figsize=(10, 8))
        plt.scatter(pred_emb[:, 0], pred_emb[:, 1], c='blue', alpha=0.5, 
                   label='Pose-only Style (Student)', s=20)
        plt.scatter(true_emb[:, 0], true_emb[:, 1], c='red', alpha=0.5,
                   label='Joint-based Style (Teacher)', s=20)
        
        plt.legend()
        plt.title('Style Vector Visualization (t-SNE)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\n可视化已保存到: {save_path}")
        
    except ImportError:
        print("\n跳过分可视化 (需要 sklearn 和 matplotlib)")


def main():
    """主测试函数"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--data', type=str,
                       default='/data0/wwb_data/ygx_data/data_ygx_pose+dof/ACCAD_CMU_merged_training_data.npz')
    parser.add_argument('--batch_size', type=int, default=1024)
    args = parser.parse_args()
    
    # 加载检查点
    print(f"加载模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    
    config_dict = checkpoint['config']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = ContrastiveStyleIK(**config_dict)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 创建归一化层
    norm_layer = NormalizationLayer(
        checkpoint['pose_mean'],
        checkpoint['pose_std'],
        checkpoint['joint_mean'],
        checkpoint['joint_std']
    ).to(device)
    
    # 加载验证数据
    print("\n加载验证数据...")
    
    class TempConfig:
        data_path = args.data
        batch_size = args.batch_size
        num_frames = config_dict['num_frames']
        num_workers = 4
        pin_memory = True
    
    _, val_loader = create_windowed_dataloaders(args.data, TempConfig())
    
    # 运行测试
    print("\n" + "=" * 60)
    print("对比学习风格 IK 泛化性测试")
    print("=" * 60)
    
    # 1. 风格对齐测试
    sim, pred_styles, true_styles = test_style_alignment(model, val_loader, norm_layer, device)
    
    # 2. IK 误差对比
    err_pose, err_joint = test_ik_error_comparison(model, val_loader, norm_layer, device)
    
    # 3. 可视化
    visualize_styles(pred_styles, true_styles)
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    if sim > 0.9 and (err_pose - err_joint) / err_joint < 0.05:
        print("✓ 模型泛化性优秀！可以安全用于推理（只用 poses）")
    elif sim > 0.8 and (err_pose - err_joint) / err_joint < 0.15:
        print("✓ 模型泛化性良好，可以使用")
    else:
        print("⚠ 模型泛化性一般，建议:")
        print("  - 增加对比损失权重")
        print("  - 增加风格维度")
        print("  - 延长训练时间")


if __name__ == "__main__":
    main()

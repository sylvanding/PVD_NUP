#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 PyTorch Geometric (PyG) 库是否安装成功
"""

import sys

def test_torch():
    """测试 PyTorch"""
    print("=" * 50)
    print("测试 PyTorch...")
    try:
        import torch
        print(f"  ✓ PyTorch 版本: {torch.__version__}")
        print(f"  ✓ CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  ✓ CUDA 版本: {torch.version.cuda}")
            print(f"  ✓ GPU 设备: {torch.cuda.get_device_name(0)}")
        return True
    except ImportError as e:
        print(f"  ✗ PyTorch 导入失败: {e}")
        return False

def test_torch_scatter():
    """测试 torch-scatter"""
    print("\n" + "=" * 50)
    print("测试 torch-scatter...")
    try:
        import torch_scatter
        print(f"  ✓ torch-scatter 版本: {torch_scatter.__version__}")
        
        # 简单功能测试
        import torch
        src = torch.randn(10, 3)
        index = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
        out = torch_scatter.scatter_mean(src, index, dim=0)
        print(f"  ✓ scatter_mean 测试通过, 输出形状: {out.shape}")
        return True
    except ImportError as e:
        print(f"  ✗ torch-scatter 导入失败: {e}")
        return False
    except Exception as e:
        print(f"  ✗ torch-scatter 功能测试失败: {e}")
        return False

def test_torch_sparse():
    """测试 torch-sparse"""
    print("\n" + "=" * 50)
    print("测试 torch-sparse...")
    try:
        import torch_sparse
        print(f"  ✓ torch-sparse 版本: {torch_sparse.__version__}")
        
        # 简单功能测试
        import torch
        index = torch.tensor([[0, 0, 1, 2], [0, 1, 1, 0]])
        value = torch.randn(4)
        adj = torch_sparse.SparseTensor(row=index[0], col=index[1], value=value, sparse_sizes=(3, 3))
        print(f"  ✓ SparseTensor 创建成功, 形状: {adj.sizes()}")
        return True
    except ImportError as e:
        print(f"  ✗ torch-sparse 导入失败: {e}")
        return False
    except Exception as e:
        print(f"  ✗ torch-sparse 功能测试失败: {e}")
        return False

def test_torch_cluster():
    """测试 torch-cluster"""
    print("\n" + "=" * 50)
    print("测试 torch-cluster...")
    try:
        import torch_cluster
        print(f"  ✓ torch-cluster 版本: {torch_cluster.__version__}")
        
        # 简单功能测试
        import torch
        x = torch.randn(100, 3)
        batch = torch.zeros(100, dtype=torch.long)
        edge_index = torch_cluster.knn_graph(x, k=6, batch=batch)
        print(f"  ✓ knn_graph 测试通过, 边数: {edge_index.shape[1]}")
        return True
    except ImportError as e:
        print(f"  ✗ torch-cluster 导入失败: {e}")
        return False
    except Exception as e:
        print(f"  ✗ torch-cluster 功能测试失败: {e}")
        return False

def test_torch_spline_conv():
    """测试 torch-spline-conv"""
    print("\n" + "=" * 50)
    print("测试 torch-spline-conv...")
    try:
        import torch_spline_conv
        print(f"  ✓ torch-spline-conv 版本: {torch_spline_conv.__version__}")
        return True
    except ImportError as e:
        print(f"  ✗ torch-spline-conv 导入失败: {e}")
        return False

def test_torch_geometric():
    """测试 torch-geometric"""
    print("\n" + "=" * 50)
    print("测试 torch-geometric...")
    try:
        import torch_geometric
        print(f"  ✓ torch-geometric 版本: {torch_geometric.__version__}")
        
        # 测试基本模块
        from torch_geometric.data import Data
        from torch_geometric.nn import GCNConv
        import torch
        
        # 创建一个简单的图数据
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        x = torch.randn(3, 16)
        data = Data(x=x, edge_index=edge_index)
        print(f"  ✓ Data 对象创建成功: {data}")
        
        # 测试 GCN 层
        conv = GCNConv(16, 32)
        out = conv(data.x, data.edge_index)
        print(f"  ✓ GCNConv 测试通过, 输出形状: {out.shape}")
        
        return True
    except ImportError as e:
        print(f"  ✗ torch-geometric 导入失败: {e}")
        return False
    except Exception as e:
        print(f"  ✗ torch-geometric 功能测试失败: {e}")
        return False


def main():
    print("\n" + "#" * 50)
    print("#    PyTorch Geometric (PyG) 安装测试脚本    #")
    print("#" * 50)
    
    results = {}
    
    # 测试各个组件
    results['PyTorch'] = test_torch()
    results['torch-scatter'] = test_torch_scatter()
    results['torch-sparse'] = test_torch_sparse()
    results['torch-cluster'] = test_torch_cluster()
    results['torch-spline-conv'] = test_torch_spline_conv()
    results['torch-geometric'] = test_torch_geometric()
    
    # 打印总结
    print("\n" + "=" * 50)
    print("测试结果总结:")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("🎉 所有测试通过! PyG 安装成功!")
    else:
        print("⚠️  部分测试失败，请检查安装。")
        print("\n提示: 可以使用以下命令安装 PyG 相关库:")
        print("  pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric")
        print("  或访问: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

# Quick Start - Molecular GNN Classification
# Save this as 'molecular_gnn_quickstart.py' and run it

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt

# Quick dataset exploration
def explore_molecular_data():
    """Quick exploration of molecular dataset"""
    
    print("🧬 Loading MUTAG molecular dataset...")
    dataset = TUDataset(root='data/MUTAG', name='MUTAG')
    
    print(f"Dataset: {dataset}")
    print(f"Number of molecules: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Number of atom features: {dataset.num_node_features}")
    
    # Look at first molecule
    molecule = dataset[0]
    print(f"\nFirst molecule:")
    print(f"- Number of atoms: {molecule.x.shape[0]}")
    print(f"- Number of bonds: {molecule.edge_index.shape[1]}")
    print(f"- Atom features shape: {molecule.x.shape}")
    print(f"- Is mutagenic: {bool(molecule.y.item())}")
    print(f"- Raw label: {molecule.y.item()}")
    
    # Analyze dataset distribution
    labels = [data.y.item() for data in dataset]
    mutagenic_count = sum(labels)
    non_mutagenic_count = len(labels) - mutagenic_count
    
    print(f"\nDataset distribution:")
    print(f"- Mutagenic molecules: {mutagenic_count} ({mutagenic_count/len(labels)*100:.1f}%)")
    print(f"- Non-mutagenic molecules: {non_mutagenic_count} ({non_mutagenic_count/len(labels)*100:.1f}%)")
    
    return dataset

if __name__ == "__main__":
    # Test if everything is working
    try:
        dataset = explore_molecular_data()
        print("✅ Setup successful! Ready to build your GNN.")
        print("\nNext steps:")
        print("1. Run the complete project code")
        print("2. Experiment with different datasets (PROTEINS, ENZYMES)")
        print("3. Try different model architectures")
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install torch torch-geometric matplotlib")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Check your installation and try again.")
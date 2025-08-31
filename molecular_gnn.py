"""
Advanced Graph Neural Network for Molecular Classification
A comprehensive implementation for drug discovery and molecular property prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.transforms import Compose, NormalizeFeatures
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class MolecularDatasetAnalyzer:
    """Comprehensive analyzer for molecular datasets"""
    
    def __init__(self, dataset_name='MUTAG'):
        """
        Initialize with dataset
        MUTAG: Mutagenic aromatic compounds (188 molecules)
        PROTEINS: Proteins dataset (1113 proteins)  
        ENZYMES: Enzyme dataset (600 enzymes)
        """
        print(f"Loading {dataset_name} dataset...")
        self.dataset = TUDataset(root=f'data/{dataset_name}', name=dataset_name)
        self.dataset_name = dataset_name
        
    def analyze_dataset(self):
        """Comprehensive dataset analysis"""
        print("="*60)
        print(f"DATASET ANALYSIS: {self.dataset_name}")
        print("="*60)
        
        # Basic statistics
        print(f"Number of graphs: {len(self.dataset)}")
        print(f"Number of classes: {self.dataset.num_classes}")
        print(f"Number of node features: {self.dataset.num_node_features}")
        print(f"Number of edge features: {self.dataset.num_edge_features}")
        
        # Analyze first graph as example
        data = self.dataset[0]
        print(f"\nExample molecule structure:")
        print(f"- Nodes (atoms): {data.x.shape[0]}")
        print(f"- Edges (bonds): {data.edge_index.shape[1]}")
        print(f"- Node features shape: {data.x.shape}")
        print(f"- Class label: {data.y.item()}")
        
        # Class distribution
        labels = [data.y.item() for data in self.dataset]
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\nClass distribution:")
        for class_idx, count in zip(unique, counts):
            print(f"  Class {class_idx}: {count} molecules ({count/len(labels)*100:.1f}%)")
            
        # Graph size statistics
        node_counts = [data.x.shape[0] for data in self.dataset]
        edge_counts = [data.edge_index.shape[1] for data in self.dataset]
        
        print(f"\nMolecule size statistics:")
        print(f"  Average atoms per molecule: {np.mean(node_counts):.1f}")
        print(f"  Average bonds per molecule: {np.mean(edge_counts):.1f}")
        print(f"  Molecule size range: {min(node_counts)} - {max(node_counts)} atoms")
        
        return self.dataset

class MolecularGNN(nn.Module):
    """
    Advanced Graph Neural Network for Molecular Classification
    Uses Graph Attention Network (GAT) layers for better performance
    """
    
    def __init__(self, num_features, num_classes, hidden_dim=64, dropout=0.2):
        super(MolecularGNN, self).__init__()
        
        # Graph Attention layers - these learn which atoms/bonds are most important
        self.conv1 = GATConv(num_features, hidden_dim, heads=8, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * 8, hidden_dim, heads=8, dropout=dropout)
        self.conv3 = GATConv(hidden_dim * 8, hidden_dim, heads=1, dropout=dropout)
        
        # Batch normalization for stable training
        self.bn1 = nn.BatchNorm1d(hidden_dim * 8)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 8)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 because we concatenate mean and max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x, edge_index, batch):
        """
        Forward pass
        x: node features [num_nodes, num_features]
        edge_index: edge connections [2, num_edges]
        batch: batch assignment for each node
        """
        
        # First GAT layer + activation + normalization
        x = self.conv1(x, edge_index)
        x = F.elu(x)  # ELU works well with GAT
        x = self.bn1(x)
        x = self.dropout(x)
        
        # Second GAT layer
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.bn2(x)
        x = self.dropout(x)
        
        # Third GAT layer
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        x = self.bn3(x)
        
        # Global pooling - convert node-level features to graph-level
        # We use both mean and max pooling for richer representation
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # Final classification
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

class MolecularTrainer:
    """Complete training pipeline for molecular GNN"""
    
    def __init__(self, dataset, model, device='cpu'):
        self.dataset = dataset
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.train_accs = []
        self.val_accs = []
        
    def prepare_data(self, train_ratio=0.8, batch_size=32):
        """Split dataset and create data loaders"""
        
        # Shuffle dataset
        dataset = self.dataset.shuffle()
        
        # Split into train/test
        train_size = int(train_ratio * len(dataset))
        train_dataset = dataset[:train_size]
        test_dataset = dataset[train_size:]
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Training molecules: {len(train_dataset)}")
        print(f"Testing molecules: {len(test_dataset)}")
        
        return self.train_loader, self.test_loader
    
    def train_epoch(self, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in self.train_loader:
            batch = batch.to(self.device)
            optimizer.zero_grad()
            
            # Forward pass
            out = self.model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
            
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self):
        """Evaluate on test set"""
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
        
        accuracy = correct / total
        return accuracy, all_preds, all_labels
    
    def train(self, num_epochs=200, lr=0.01, weight_decay=1e-4):
        """Complete training pipeline"""
        
        # Optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.NLLLoss()  # Negative log-likelihood for log_softmax output
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        
        print("Starting training...")
        print("="*50)
        
        best_val_acc = 0
        for epoch in tqdm(range(num_epochs), desc="Training"):
            
            # Train
            train_loss, train_acc = self.train_epoch(optimizer, criterion)
            
            # Evaluate
            val_acc, _, _ = self.evaluate()
            
            # Update learning rate
            scheduler.step()
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_molecular_gnn.pth')
            
            # Print progress
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | "
                      f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        print(f"\nBest validation accuracy: {best_val_acc:.4f}")
        return best_val_acc

    def plot_training_curves(self):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curve
        ax1.plot(self.train_losses, label='Training Loss', color='red', alpha=0.7)
        ax1.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(self.train_accs, label='Training Accuracy', color='blue', alpha=0.7)
        ax2.plot(self.val_accs, label='Validation Accuracy', color='green', alpha=0.7)
        ax2.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

    def detailed_evaluation(self):
        """Comprehensive model evaluation with metrics and visualizations"""
        
        # Load best model
        self.model.load_state_dict(torch.load('best_molecular_gnn.pth'))
        
        # Get predictions
        accuracy, predictions, labels = self.evaluate()
        
        print("="*60)
        print("FINAL MODEL EVALUATION")
        print("="*60)
        print(f"Final Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(labels, predictions))
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Molecular Classification', fontsize=16, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy

# Main execution pipeline
def main():
    """Main execution function"""
    
    print("🧬 MOLECULAR GRAPH NEURAL NETWORK PROJECT 🧬")
    print("=" * 60)
    
    # Step 1: Data Analysis
    analyzer = MolecularDatasetAnalyzer('MUTAG')  # You can try 'PROTEINS', 'ENZYMES' too
    dataset = analyzer.analyze_dataset()
    
    # Step 2: Model Creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = MolecularGNN(
        num_features=dataset.num_node_features,
        num_classes=dataset.num_classes,
        hidden_dim=64,
        dropout=0.3
    )
    
    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Step 3: Training Pipeline
    trainer = MolecularTrainer(dataset, model, device)
    train_loader, test_loader = trainer.prepare_data(batch_size=32)
    
    # Step 4: Train the model
    best_accuracy = trainer.train(num_epochs=200, lr=0.01)
    
    # Step 5: Visualize results
    trainer.plot_training_curves()
    
    # Step 6: Final evaluation
    final_accuracy = trainer.detailed_evaluation()
    
    print(f"\n🎉 PROJECT COMPLETED! 🎉")
    print(f"Final Model Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print(f"Model saved as: 'best_molecular_gnn.pth'")
    print(f"Visualizations saved as: 'training_curves.png' and 'confusion_matrix.png'")

if __name__ == "__main__":
    main()
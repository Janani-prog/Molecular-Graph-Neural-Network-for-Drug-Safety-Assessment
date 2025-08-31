
# Production Deployment Example
from flask import Flask, request, jsonify
from rdkit import Chem
import torch

app = Flask(__name__)
model = torch.load('best_molecular_gnn.pth')

@app.route('/predict', methods=['POST'])
def predict_toxicity():
    """API endpoint for toxicity prediction"""
    
    smiles = request.json['smiles']  # Chemical structure string
    
    # Convert SMILES to molecular graph
    mol = Chem.MolFromSmiles(smiles)
    graph_data = smiles_to_graph(mol)
    
    # Predict with GNN
    with torch.no_grad():
        prediction = model(graph_data)
        toxicity_prob = torch.softmax(prediction, dim=1)[0][1].item()
    
    return jsonify({
        'molecule': smiles,
        'toxicity_probability': toxicity_prob,
        'risk_level': 'HIGH' if toxicity_prob > 0.7 else 'LOW'
    })

# Usage: curl -X POST -H "Content-Type: application/json" 
#        -d '{"smiles":"CCO"}' http://localhost:5000/predict
        
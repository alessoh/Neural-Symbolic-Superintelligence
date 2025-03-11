import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
from tabulate import tabulate
import random
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from tqdm import tqdm
import time
import pandas as pd
from collections import defaultdict

# Set random seeds for reproducibility and 
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# ============================
# 1. Enhanced Neural Component
# ============================
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        d_k = k.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply attention
        output, attn_weights = self.attention(q, k, v, mask)
        
        # Concatenate heads and put through final linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_linear(output)
        
        return output, attn_weights

class PositionwiseFeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self attention
        residual = x
        x = self.norm1(x)
        x_attn, attn_weights = self.self_attn(x, x, x, mask)
        x = residual + self.dropout(x_attn)
        
        # Feed forward
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        
        return x, attn_weights

class AdvancedNeuralModel(nn.Module):
    def __init__(self, input_dim, num_classes, embed_dim=128, num_layers=3, num_heads=8, ff_dim=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        
    def forward(self, x):
        # Convert to batch_size x 1 x input_dim and embed
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = self.embedding(x)
        
        # Pass through transformer layers
        attentions = []
        for layer in self.transformer_layers:
            x, attn = layer(x)
            attentions.append(attn)
            
        # Use the representation of the first token for classification
        x = x.squeeze(1) if x.size(1) == 1 else x[:, 0]
        
        # Classify
        logits = self.classifier(x)
        
        return logits, x, attentions

# ===========================
# 2. Enhanced Symbolic Component with Knowledge Graph
# ===========================
class EnhancedKnowledgeGraph:
    def __init__(self):
        self.entities = {}                 # entity_id -> name
        self.relations = []                # (source_id, relation_type, target_id, weight)
        self.rules = []                    # (premise_ids, conclusion_id, confidence)
        self.hierarchy = defaultdict(set)  # entity_id -> set of parent entity_ids
        self.entity_attrs = {}             # entity_id -> {attribute: value}
        
    def add_entity(self, entity_id, name, attributes=None):
        """Add an entity to the knowledge graph with optional attributes"""
        self.entities[entity_id] = name
        if attributes:
            self.entity_attrs[entity_id] = attributes
        return self
        
    def add_relation(self, source_id, relation_type, target_id, weight=1.0):
        """Add a relation between two entities with a weight"""
        self.relations.append((source_id, relation_type, target_id, weight))
        return self
        
    def add_rule(self, premise_ids, conclusion_id, confidence=1.0):
        """Add a logical rule with a confidence score"""
        self.rules.append((premise_ids, conclusion_id, confidence))
        return self
        
    def add_hierarchy(self, child_id, parent_id):
        """Add hierarchical relationship (e.g., specific_fever is a fever)"""
        self.hierarchy[child_id].add(parent_id)
        return self
    
    def get_ancestors(self, entity_id):
        """Get all ancestors of an entity in the hierarchy"""
        ancestors = set()
        to_process = list(self.hierarchy[entity_id])
        
        while to_process:
            parent = to_process.pop()
            ancestors.add(parent)
            to_process.extend(self.hierarchy[parent] - ancestors)
            
        return ancestors
        
    def reason(self, active_entities, max_hops=3):
        """
        Apply enhanced reasoning to derive new knowledge
        
        Args:
            active_entities: Set of currently active entity IDs
            max_hops: Maximum number of reasoning hops
            
        Returns:
            inferred: Set of inferred entity IDs
            reasoning_steps: Dictionary of reasoning steps for each entity
            confidences: Dictionary of confidence values for each entity
            class_scores: Dictionary of confidence scores for each class
        """
        # Initialize with active entities and their hierarchical parents
        inferred = set(active_entities)
        for entity in list(active_entities):
            inferred.update(self.get_ancestors(entity))
            
        reasoning_steps = {}
        confidences = {}
        
        # Default class scores (usually 4 classes: cold, flu, covid, pneumonia)
        class_scores = defaultdict(float)
        
        # Initialize reasoning steps and confidences for active entities
        for entity_id in active_entities:
            if entity_id in self.entities:
                reasoning_steps[entity_id] = f"Given: {self.entities[entity_id]}"
                confidences[entity_id] = 1.0
        
        # Add reasoning steps for ancestor entities
        for entity_id in inferred - set(active_entities):
            if entity_id in self.entities:
                for child in active_entities:
                    if entity_id in self.get_ancestors(child):
                        reasoning_steps[entity_id] = f"Hierarchical: {self.entities[child]} is a type of {self.entities[entity_id]}"
                        confidences[entity_id] = 0.95  # High confidence for hierarchical relationships
                        break
        
        # Multi-hop reasoning
        for _ in range(max_hops):
            new_inferences = set()
            
            # Apply relations
            for source_id, relation_type, target_id, weight in self.relations:
                if source_id in inferred and target_id not in inferred:
                    new_inferences.add(target_id)
                    step = f"{self.entities[source_id]} --{relation_type}--> {self.entities[target_id]}"
                    reasoning_steps[target_id] = step
                    confidences[target_id] = weight * confidences.get(source_id, 1.0)
                    
                    # Update class scores directly if target is a class
                    # Assuming classes have IDs 0, 1, 2, 3, etc.
                    if target_id < 10:  # Adjust this threshold based on your class IDs
                        class_scores[target_id] = max(class_scores[target_id], confidences[target_id])
            
            # Apply rules
            for premise_ids, conclusion_id, confidence in self.rules:
                if all(p_id in inferred for p_id in premise_ids) and conclusion_id not in inferred:
                    new_inferences.add(conclusion_id)
                    premises = [self.entities[p_id] for p_id in premise_ids]
                    step = f"Rule: IF {' AND '.join(premises)} THEN {self.entities[conclusion_id]}"
                    reasoning_steps[conclusion_id] = step
                    
                    # Calculate rule confidence based on premises
                    premise_conf = min([confidences.get(p_id, 1.0) for p_id in premise_ids])
                    rule_conf = confidence * premise_conf
                    confidences[conclusion_id] = rule_conf
                    
                    # Update class scores
                    if conclusion_id < 10:  # Adjust for class IDs
                        class_scores[conclusion_id] = max(class_scores[conclusion_id], rule_conf)
            
            # If no new inferences were made, stop
            if not new_inferences:
                break
                
            inferred.update(new_inferences)
        
        # Add further confidence adjustments for comorbidities or risk factors
        for entity_id in inferred:
            attrs = self.entity_attrs.get(entity_id, {})
            if 'risk_factor' in attrs and attrs['risk_factor'] > 0:
                for class_id, score in class_scores.items():
                    # Increase class score based on risk factors
                    if attrs.get(f'increases_{class_id}', 0) > 0:
                        multiplier = 1 + (attrs['risk_factor'] * attrs[f'increases_{class_id}'])
                        class_scores[class_id] = min(0.99, score * multiplier)
                        reasoning_steps[f"risk_{entity_id}_{class_id}"] = (
                            f"Risk Factor: {self.entities[entity_id]} increases likelihood of "
                            f"{self.entities[class_id]} by {multiplier:.1f}x"
                        )
        
        return inferred, reasoning_steps, confidences, dict(class_scores)

# ===========================
# A3. Advanced Neural-Symbolic Interface
# ===========================
class AdvancedNeuralSymbolicInterface(nn.Module):
    def __init__(self, hidden_dim, num_symbols, num_classes):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_symbols = num_symbols
        self.num_classes = num_classes
        
        # Neural to symbol mapping
        self.neural_to_symbol = nn.Linear(hidden_dim, num_symbols)
        
        # Symbol to class mapping with learnable weights
        self.symbol_to_class = nn.Parameter(torch.zeros(num_symbols, num_classes))
        
        # Adaptive threshold parameters
        self.threshold_base = nn.Parameter(torch.ones(1) * 0.5)
        self.threshold_scale = nn.Parameter(torch.ones(num_symbols) * 0.1)
        
    def forward(self, neural_repr):
        """Forward pass for training"""
        symbol_logits = self.neural_to_symbol(neural_repr)
        return symbol_logits
    
    def get_thresholds(self):
        """Get adaptive thresholds for each symbol"""
        return torch.clamp(self.threshold_base + self.threshold_scale, 0.1, 0.9)
    
    def neural_to_symbolic(self, neural_repr):
        """Convert neural representations to symbolic activations with adaptive thresholds"""
        symbol_logits = self.neural_to_symbol(neural_repr)
        symbol_probs = torch.sigmoid(symbol_logits)
        
        thresholds = self.get_thresholds()
        activations = (symbol_probs > thresholds).float()
        
        return activations, symbol_probs, symbol_logits
    
    def symbolic_to_neural_prediction(self, symbolic_activations, confidences=None):
        """Convert symbolic activations to class predictions"""
        if confidences is None:
            # Simple matrix multiplication
            class_scores = torch.matmul(symbolic_activations, self.symbol_to_class)
        else:
            # Weight by confidences
            conf_tensor = torch.zeros_like(symbolic_activations)
            for i, confs in enumerate(confidences):
                for symbol_id, conf in confs.items():
                    if isinstance(symbol_id, int) and symbol_id < conf_tensor.shape[1]:
                        conf_tensor[i, symbol_id] = conf
            
            weighted_activations = symbolic_activations * conf_tensor
            class_scores = torch.matmul(weighted_activations, self.symbol_to_class)
        
        return class_scores
    
    def set_symbol_to_class_mapping(self, symbol_to_class_dict):
        """Set initial values for symbol to class mapping"""
        with torch.no_grad():
            for symbol_id, class_weights in symbol_to_class_dict.items():
                for class_id, weight in class_weights.items():
                    self.symbol_to_class[symbol_id, class_id] = weight

# ===========================
# 4. Advanced Metacognitive Control
# ===========================
class AdvancedMetacognitiveController:
    def __init__(self, neural_threshold=0.85, symbolic_threshold=0.75, learning_rate=0.01):
        self.neural_threshold = neural_threshold
        self.symbolic_threshold = symbolic_threshold
        self.learning_rate = learning_rate
        self.strategy_history = []
        self.correct_strategy_counts = {'neural': 0, 'symbolic': 0, 'hybrid': 0}
        
    def update_thresholds(self, neural_correct, symbolic_correct, strategy):
        """Update thresholds based on which strategy was correct"""
        # Only update if one was correct and one was wrong
        if neural_correct != symbolic_correct:
            if neural_correct:
                # Neural was right, symbolic was wrong - favor neural more
                self.neural_threshold = max(0.7, self.neural_threshold - self.learning_rate)
                self.symbolic_threshold = min(0.9, self.symbolic_threshold + self.learning_rate)
                self.correct_strategy_counts['neural'] += 1
            else:
                # Symbolic was right, neural was wrong - favor symbolic more
                self.neural_threshold = min(0.9, self.neural_threshold + self.learning_rate)
                self.symbolic_threshold = max(0.7, self.symbolic_threshold - self.learning_rate)
                self.correct_strategy_counts['symbolic'] += 1
        elif neural_correct and symbolic_correct:
            # Both were correct
            if strategy == 'hybrid':
                self.correct_strategy_counts['hybrid'] += 1
        
    def decide_strategy(self, neural_conf, symbolic_conf, risk_level='medium'):
        """Decide which strategy to use based on confidence levels and risk"""
        neural_threshold = self.neural_threshold
        symbolic_threshold = self.symbolic_threshold
        
        # Adjust thresholds based on risk level
        if risk_level == 'high':
            # For high-risk patients, be more cautious with neural and favor symbolic
            neural_threshold += 0.1
            symbolic_threshold -= 0.1
        elif risk_level == 'low':
            # For low-risk, favor faster neural processing
            neural_threshold -= 0.1
            symbolic_threshold += 0.1
            
        if neural_conf >= neural_threshold and symbolic_conf < symbolic_threshold:
            strategy = {
                'strategy': 'neural',
                'neural_weight': 1.0,
                'symbolic_weight': 0.0,
                'explanation': f'Using neural prediction (high confidence: {neural_conf:.2f})'
            }
        elif symbolic_conf >= symbolic_threshold and neural_conf < neural_threshold:
            strategy = {
                'strategy': 'symbolic',
                'neural_weight': 0.0,
                'symbolic_weight': 1.0,
                'explanation': f'Using symbolic reasoning (high confidence: {symbolic_conf:.2f})'
            }
        else:
            # Weighted combination proportional to confidence
            total_conf = neural_conf + symbolic_conf
            neural_weight = neural_conf / total_conf if total_conf > 0 else 0.5
            symbolic_weight = 1.0 - neural_weight
            
            strategy = {
                'strategy': 'hybrid',
                'neural_weight': neural_weight,
                'symbolic_weight': symbolic_weight,
                'explanation': (f'Using weighted combination based on confidence '
                                f'(neural: {neural_weight:.2f}, symbolic: {symbolic_weight:.2f})')
            }
        
        self.strategy_history.append(strategy['strategy'])
        return strategy
    
    def get_strategy_stats(self):
        """Get statistics on strategy usage"""
        if not self.strategy_history:
            return {'neural': 0, 'symbolic': 0, 'hybrid': 0}
            
        return {
            'neural': self.strategy_history.count('neural') / len(self.strategy_history),
            'symbolic': self.strategy_history.count('symbolic') / len(self.strategy_history),
            'hybrid': self.strategy_history.count('hybrid') / len(self.strategy_history),
            'correct_neural': self.correct_strategy_counts['neural'],
            'correct_symbolic': self.correct_strategy_counts['symbolic'],
            'correct_hybrid': self.correct_strategy_counts['hybrid'],
        }

# ===========================
# 5. Enhanced NEXUS Integrated Model
# ===========================
class EnhancedNEXUSModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_symbols, symbol_names, class_names, 
                 embed_dim=128, device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_symbols = num_symbols
        self.symbol_names = symbol_names
        self.class_names = class_names
        self.symbol_to_id = {name: i for i, name in enumerate(symbol_names)}
        self.device = device
        
        # Move to specified device
        self = self.to(device)
        
        # Neural model
        self.neural_model = AdvancedNeuralModel(
            input_dim=input_dim, 
            num_classes=num_classes,
            embed_dim=embed_dim
        ).to(device)
        
        # Knowledge graph
        self.knowledge_graph = EnhancedKnowledgeGraph()
        
        # Neural-symbolic interface
        self.interface = AdvancedNeuralSymbolicInterface(
            hidden_dim=embed_dim,
            num_symbols=num_symbols,
            num_classes=num_classes
        ).to(device)
        
        # Metacognitive controller
        self.metacognitive = AdvancedMetacognitiveController()
        
        # Evaluation results tracking
        self.eval_results = {
            'neural': {'correct': 0, 'total': 0, 'confusion': None, 'predictions': [], 'true_labels': [], 'confidence': []},
            'symbolic': {'correct': 0, 'total': 0, 'confusion': None, 'predictions': [], 'true_labels': [], 'confidence': []},
            'nexus': {'correct': 0, 'total': 0, 'confusion': None, 'predictions': [], 'true_labels': [], 'confidence': []}
        }
        
        # Case tracker for detailed analysis
        self.case_details = []
        
    def init_knowledge_graph(self):
        """Initialize the knowledge graph with entities (symptoms, conditions, etc.)"""
        kg = self.knowledge_graph
        
        # Add entities (symptoms and conditions)
        for i, name in enumerate(self.symbol_names):
            kg.add_entity(i, name)
            
        # Add classes (medical conditions)
        for i, name in enumerate(self.class_names):
            kg.add_entity(i, name)
            
        return kg
    
    def forward(self, x):
        """Forward pass for training"""
        x = x.to(self.device)
        return self.neural_model(x)[0]
    
    def train_neural(self, dataloader, num_epochs=5, lr=0.001, scheduler=None, weight_decay=1e-5):
        """Train the neural component using the provided dataloader"""
        self.neural_model.train()
        optimizer = torch.optim.AdamW(
            self.neural_model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        if scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=num_epochs * len(dataloader)
            )
        elif scheduler == 'reduce':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5, 
                patience=2
            )
        
        criterion = nn.CrossEntropyLoss()
        
        epoch_stats = []
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            
            # Use tqdm for progress tracking
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs, _, _ = self.neural_model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                    scheduler.step()
                
                # Track statistics
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                epoch_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100 * correct / total:.2f}%"
                })
            
            avg_loss = epoch_loss / len(dataloader)
            accuracy = 100 * correct / total
            
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            epoch_stats.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'accuracy': accuracy
            })
            
        return epoch_stats
    
    def diagnose(self, x, active_symptoms=None, risk_level='medium'):
        """
        Diagnose a patient based on symptoms using neural and symbolic components
        
        Args:
            x: Input features tensor
            active_symptoms: List of symptom names (optional, for symbolic reasoning)
            risk_level: Risk level of the patient (low, medium, high)
            
        Returns:
            Dictionary with neural, symbolic, and NEXUS predictions
        """
        self.neural_model.eval()
        self.interface.eval()
        
        # Convert to tensor if necessary
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        x = x.to(self.device)
        
        with torch.no_grad():
            # Neural processing
            neural_logits, neural_repr, _ = self.neural_model(x)
            neural_probs = F.softmax(neural_logits, dim=1)
            neural_pred = torch.argmax(neural_probs, dim=1).item()
            neural_conf = neural_probs[0, neural_pred].item()
            
            # Neural-to-symbolic translation
            symbolic_activations, similarities, _ = self.interface.neural_to_symbolic(neural_repr)
            
            # If active symptoms provided, use those instead of derived ones
            if active_symptoms is not None:
                symptom_ids = [self.symbol_to_id[name] for name in active_symptoms if name in self.symbol_to_id]
            else:
                # Extract activated symptoms from neural representations
                symptom_ids = torch.nonzero(symbolic_activations[0]).squeeze(-1).tolist()
                if not isinstance(symptom_ids, list):
                    symptom_ids = [symptom_ids]
            
            # Symbolic reasoning
            inferred_ids, reasoning_steps, confidences, class_scores = self.knowledge_graph.reason(symptom_ids)
            
            # Convert class scores to tensor and normalize
            symbolic_scores = torch.zeros(1, self.num_classes, device=self.device)
            for class_id, score in class_scores.items():
                if class_id < self.num_classes:
                    symbolic_scores[0, class_id] = score
                    
            # If all scores are zero, set equal probabilities
            if symbolic_scores.sum() == 0:
                symbolic_probs = torch.ones(1, self.num_classes, device=self.device) / self.num_classes
            else:
                symbolic_probs = F.softmax(symbolic_scores, dim=1)
                
            symbolic_pred = torch.argmax(symbolic_probs, dim=1).item()
            symbolic_conf = symbolic_probs[0, symbolic_pred].item()
            
            # Metacognitive control
            strategy = self.metacognitive.decide_strategy(neural_conf, symbolic_conf, risk_level)
            
            # Final prediction based on strategy
            if strategy['strategy'] == 'neural':
                final_pred = neural_pred
                final_conf = neural_conf
            elif strategy['strategy'] == 'symbolic':
                final_pred = symbolic_pred
                final_conf = symbolic_conf
            else:  # hybrid
                combined_probs = (
                    strategy['neural_weight'] * neural_probs + 
                    strategy['symbolic_weight'] * symbolic_probs
                )
                final_pred = torch.argmax(combined_probs, dim=1).item()
                final_conf = combined_probs[0, final_pred].item()
        
        # Create result dictionary
        result = {
            'neural': {
                'prediction': neural_pred,
                'confidence': neural_conf,
                'class_name': self.class_names[neural_pred],
                'probabilities': neural_probs[0].cpu().numpy()
            },
            'symbolic': {
                'prediction': symbolic_pred,
                'confidence': symbolic_conf,
                'class_name': self.class_names[symbolic_pred],
                'reasoning_steps': reasoning_steps,
                'inferred_symbols': [self.symbol_names[i] for i in inferred_ids 
                                     if i < len(self.symbol_names)],
                'active_symptoms': [self.symbol_names[i] for i in symptom_ids 
                                   if i < len(self.symbol_names)],
                'class_scores': class_scores,
                'probabilities': symbolic_probs[0].cpu().numpy()
            },
            'nexus': {
                'prediction': final_pred,
                'confidence': final_conf,
                'class_name': self.class_names[final_pred],
                'strategy': strategy
            }
        }
        
        return result
    
    def evaluate(self, dataloader, symptom_dict=None, feedback=True):
        """
        Evaluate the model on a test set
        
        Args:
            dataloader: DataLoader with test data
            symptom_dict: Dictionary mapping indices to active symptoms
            feedback: Whether to provide feedback to metacognitive controller
            
        Returns:
            Dictionary with evaluation results
        """
        self.neural_model.eval()
        self.interface.eval()
        
        # Reset evaluation results
        for key in self.eval_results:
            self.eval_results[key]['correct'] = 0
            self.eval_results[key]['total'] = 0
            self.eval_results[key]['predictions'] = []
            self.eval_results[key]['true_labels'] = []
            self.eval_results[key]['confidence'] = []
            
        self.eval_results['neural']['confusion'] = np.zeros((self.num_classes, self.num_classes))
        self.eval_results['symbolic']['confusion'] = np.zeros((self.num_classes, self.num_classes))
        self.eval_results['nexus']['confusion'] = np.zeros((self.num_classes, self.num_classes))
        
        # Track agreement between methods
        agreement_cases = {
            'all_correct': 0, 
            'all_wrong': 0, 
            'neural_only': 0, 
            'symbolic_only': 0, 
            'nexus_better': 0
        }
        
        # Use tqdm for progress tracking
        progress_bar = tqdm(dataloader, desc="Evaluating")
        
        # Process each batch
        self.case_details = []  # Reset case details
        
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(progress_bar):
                # Get active symptoms if provided
                active_symptoms = symptom_dict.get(i, None) if symptom_dict else None
                
                # Get diagnoses
                result = self.diagnose(inputs, active_symptoms)
                true_label = labels.item()
                
                # Track case details for analysis
                case_detail = {
                    'index': i,
                    'true_label': true_label,
                    'true_class': self.class_names[true_label],
                    'neural_pred': result['neural']['prediction'],
                    'neural_conf': result['neural']['confidence'],
                    'symbolic_pred': result['symbolic']['prediction'],
                    'symbolic_conf': result['symbolic']['confidence'],
                    'nexus_pred': result['nexus']['prediction'],
                    'nexus_conf': result['nexus']['confidence'],
                    'nexus_strategy': result['nexus']['strategy']['strategy'],
                    'active_symptoms': active_symptoms
                }
                self.case_details.append(case_detail)
                
                # Update results for each component
                for key in ['neural', 'symbolic', 'nexus']:
                    pred = result[key]['prediction']
                    conf = result[key]['confidence']
                    
                    # Update confusion matrix
                    self.eval_results[key]['confusion'][true_label, pred] += 1
                    
                    # Track predictions and labels
                    self.eval_results[key]['predictions'].append(pred)
                    self.eval_results[key]['true_labels'].append(true_label)
                    self.eval_results[key]['confidence'].append(conf)
                    
                    # Count correct predictions
                    if pred == true_label:
                        self.eval_results[key]['correct'] += 1
                    self.eval_results[key]['total'] += 1
                
                # Track agreement between methods
                neural_correct = result['neural']['prediction'] == true_label
                symbolic_correct = result['symbolic']['prediction'] == true_label
                nexus_correct = result['nexus']['prediction'] == true_label
                
                if neural_correct and symbolic_correct and nexus_correct:
                    agreement_cases['all_correct'] += 1
                elif not neural_correct and not symbolic_correct and not nexus_correct:
                    agreement_cases['all_wrong'] += 1
                elif neural_correct and not symbolic_correct:
                    agreement_cases['neural_only'] += 1
                elif not neural_correct and symbolic_correct:
                    agreement_cases['symbolic_only'] += 1
                elif nexus_correct and (not neural_correct or not symbolic_correct):
                    agreement_cases['nexus_better'] += 1
                
                # Provide feedback to metacognitive controller if enabled
                if feedback:
                    self.metacognitive.update_thresholds(
                        neural_correct, 
                        symbolic_correct,
                        result['nexus']['strategy']['strategy']
                    )
            
            # Calculate accuracy for each method
            for key in self.eval_results:
                if self.eval_results[key]['total'] > 0:
                    self.eval_results[key]['accuracy'] = (
                        self.eval_results[key]['correct'] / self.eval_results[key]['total']
                    )
                else:
                    self.eval_results[key]['accuracy'] = 0
            
            # Store agreement cases
            self.eval_results['agreement_cases'] = agreement_cases
        
        return self.eval_results
    
    def explain_diagnosis(self, result, detail_level='medium', include_confidence=True):
        """
        Generate an explanation of the diagnosis at different levels of detail
        
        Args:
            result: Diagnosis result dictionary
            detail_level: 'simple', 'medium', or 'high'
            include_confidence: Whether to include confidence scores
            
        Returns:
            String with the explanation
        """
        conf_str = f" (Confidence: {result['nexus']['confidence']:.2f})" if include_confidence else ""
        explanation = [f"Diagnosis: {result['nexus']['class_name']}{conf_str}"]
        explanation.append(f"Strategy: {result['nexus']['strategy']['strategy']}")
        explanation.append(f"Reason: {result['nexus']['strategy']['explanation']}")
        
        if detail_level == 'simple':
            # Simple explanation only includes the basics
            return "\n".join(explanation)
        
        # Medium and high explanations include more details
        explanation.append("\nDetected Symptoms:")
        if 'active_symptoms' in result['symbolic'] and result['symbolic']['active_symptoms']:
            explanation.append(f"  {', '.join(result['symbolic']['active_symptoms'])}")
        else:
            explanation.append("  None detected")
        
        explanation.append("\nSymbolic Reasoning:")
        explanation.append(f"Identified concepts: {', '.join(result['symbolic']['inferred_symbols'])}")
        
        if detail_level == 'high' and result['symbolic']['reasoning_steps']:
            explanation.append("\nReasoning steps:")
            
            # Group reasoning steps by type for better organization
            symptom_steps = []
            rule_steps = []
            other_steps = []
            
            for symbol_id, step in result['symbolic']['reasoning_steps'].items():
                if isinstance(symbol_id, (int, np.int64)) and symbol_id < len(self.symbol_names) + len(self.class_names):
                    if symbol_id < len(self.symbol_names):
                        symbol_name = self.symbol_names[symbol_id]
                    else:
                        symbol_name = self.class_names[symbol_id - len(self.symbol_names)]
                        
                    formatted_step = f"- {symbol_name}: {step}"
                    
                    if "Given" in step:
                        symptom_steps.append(formatted_step)
                    elif "Rule" in step:
                        rule_steps.append(formatted_step)
                    else:
                        other_steps.append(formatted_step)
                else:
                    # Handle non-integer or special symbol IDs (like risk factors)
                    other_steps.append(f"- {step}")
            
            # Add the grouped steps with headers
            if symptom_steps:
                explanation.append("Initial symptoms:")
                explanation.extend(symptom_steps)
                
            if rule_steps:
                explanation.append("\nApplied medical rules:")
                explanation.extend(rule_steps)
                
            if other_steps:
                explanation.append("\nOther reasoning:")
                explanation.extend(other_steps)
        
        # Add model comparison
        neural_conf = f" (Confidence: {result['neural']['confidence']:.2f})" if include_confidence else ""
        symbolic_conf = f" (Confidence: {result['symbolic']['confidence']:.2f})" if include_confidence else ""
        
        explanation.append(f"\nNeural model prediction: {result['neural']['class_name']}{neural_conf}")
        explanation.append(f"Symbolic model prediction: {result['symbolic']['class_name']}{symbolic_conf}")
        
        # For high detail, add class probabilities
        if detail_level == 'high' and include_confidence:
            explanation.append("\nClass probabilities (Neural):")
            for i, prob in enumerate(result['neural']['probabilities']):
                explanation.append(f"  {self.class_names[i]}: {prob:.4f}")
                
            explanation.append("\nClass scores (Symbolic):")
            for i in range(len(self.class_names)):
                score = result['symbolic']['class_scores'].get(i, 0)
                explanation.append(f"  {self.class_names[i]}: {score:.4f}")
        
        return "\n".join(explanation)
    
    def visualize_results(self, output_prefix=None, save_figures=False, show_figures=True):
        """
        Visualize evaluation results with enhanced plots
        
        Args:
            output_prefix: Prefix for saved figures (if save_figures=True)
            save_figures: Whether to save figures to disk
            show_figures: Whether to display figures
            
        Returns:
            Dictionary with summary statistics
        """
        if self.eval_results['neural']['confusion'] is None:
            print("No evaluation results to visualize. Run evaluate() first.")
            return
        
        # Prepare results for visualization
        models = ['neural', 'symbolic', 'nexus']
        titles = ['Neural Model', 'Symbolic Model', 'NEXUS Model']
        colors = ['#3498db', '#2ecc71', '#e74c3c']  # Blue, Green, Red
        
        # 1. Accuracy Comparison
        try:
            accuracies = [
                self.eval_results[model]['accuracy'] * 100 for model in models
            ]
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(titles, accuracies, color=colors, alpha=0.8, width=0.6)
            plt.title('Accuracy Comparison', fontsize=16)
            plt.ylabel('Accuracy (%)', fontsize=14)
            plt.ylim(0, 100)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1, 
                        f"{height:.1f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            
            if save_figures and output_prefix:
                plt.savefig(f"{output_prefix}_accuracy.png", dpi=300, bbox_inches='tight')
            
            if show_figures:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"Error in accuracy plot: {e}")
        
        # 2. Confusion Matrices
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Confusion Matrices (Normalized by Row)', fontsize=16, y=1.05)
            
            for i, (model, title, color) in enumerate(zip(models, titles, colors)):
                confusion = self.eval_results[model]['confusion']
                row_sums = confusion.sum(axis=1, keepdims=True)
                norm_confusion = np.where(row_sums == 0, 0, confusion / row_sums)
                
                sns.heatmap(norm_confusion, annot=True, fmt='.2f', cmap=f"{color}_r", 
                            xticklabels=self.class_names, yticklabels=self.class_names, ax=axes[i])
                axes[i].set_title(title, fontsize=14)
                axes[i].set_ylabel('True Label' if i == 0 else '', fontsize=12)
                axes[i].set_xlabel('Predicted Label', fontsize=12)
                
            plt.tight_layout()
            
            if save_figures and output_prefix:
                plt.savefig(f"{output_prefix}_confusion.png", dpi=300, bbox_inches='tight')
            
            if show_figures:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"Error in confusion matrix plot: {e}")
        
        # 3. Model Agreement Analysis
        try:
            agreement = self.eval_results['agreement_cases']
            labels = ['All Correct', 'Neural Only', 'Symbolic Only', 'NEXUS Better', 'All Wrong']
            values = [agreement['all_correct'], agreement['neural_only'], agreement['symbolic_only'], 
                    agreement['nexus_better'], agreement['all_wrong']]
            colors = ['#27ae60', '#3498db', '#2ecc71', '#e74c3c', '#95a5a6']  # Green, Blue, Green, Red, Gray
            
            # Calculate percentages
            total_cases = sum(values)
            percentages = [100 * v / total_cases for v in values] if total_cases > 0 else [0] * 5
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            
            # Bar plot
            bars = ax1.bar(labels, values, color=colors, alpha=0.8)
            ax1.set_title('Model Agreement Analysis (Counts)', fontsize=16)
            ax1.set_ylabel('Number of Cases', fontsize=14)
            ax1.grid(axis='y', linestyle='--', alpha=0.7)
            
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5, 
                        f"{int(height)}", ha='center', va='bottom', fontsize=12)
            
            # Pie chart
            wedges, texts, autotexts = ax2.pie(values, labels=labels, autopct='%1.1f%%', 
                                            colors=colors, shadow=False, startangle=90)
            ax2.set_title('Model Agreement Analysis (Percentages)', fontsize=16)
            
            # Make the percentage text more readable
            for autotext in autotexts:
                autotext.set_fontsize(10)
                autotext.set_weight('bold')
            
            plt.tight_layout()
            
            if save_figures and output_prefix:
                plt.savefig(f"{output_prefix}_agreement.png", dpi=300, bbox_inches='tight')
            
            if show_figures:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"Error in agreement plot: {e}")
        
        # 4. Class-wise Performance
        try:
            # Calculate F1 scores for each class and model
            f1_scores = np.zeros((self.num_classes, 3))
            
            for c in range(self.num_classes):
                for i, model in enumerate(models):
                    true_labels = np.array(self.eval_results[model]['true_labels'])
                    predictions = np.array(self.eval_results[model]['predictions'])
                    
                    # Calculate precision, recall, and F1
                    tp = np.sum((predictions == c) & (true_labels == c))
                    fp = np.sum((predictions == c) & (true_labels != c))
                    fn = np.sum((predictions != c) & (true_labels == c))
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    f1_scores[c, i] = f1
            
            # Plot F1 scores
            plt.figure(figsize=(14, 8))
            x = np.arange(self.num_classes)
            width = 0.25
            
            for i, (model, color) in enumerate(zip(models, colors)):
                plt.bar(x + (i - 1) * width, f1_scores[:, i], width, color=color, label=titles[i], alpha=0.8)
            
            plt.xlabel('Class', fontsize=14)
            plt.ylabel('F1 Score', fontsize=14)
            plt.title('F1 Score by Class and Model', fontsize=16)
            plt.xticks(x, self.class_names, rotation=45, ha='right')
            plt.ylim(0, 1.0)
            plt.legend(fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            if save_figures and output_prefix:
                plt.savefig(f"{output_prefix}_f1_scores.png", dpi=300, bbox_inches='tight')
            
            if show_figures:
                plt.show()
            else:
                plt.close()
                
            # Also print the tabular results
            class_results = []
            for c in range(self.num_classes):
                class_results.append([
                    self.class_names[c], 
                    f"{f1_scores[c, 0]:.3f}", 
                    f"{f1_scores[c, 1]:.3f}", 
                    f"{f1_scores[c, 2]:.3f}"
                ])
            
            print("\nClass-wise F1 Performance:")
            print(tabulate(class_results, headers=['Class', 'Neural F1', 'Symbolic F1', 'NEXUS F1'], tablefmt='grid'))
            
        except Exception as e:
            print(f"Error in class-wise performance visualization: {e}")
        
        # 5. Confidence Distribution
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Confidence Distribution by Model', fontsize=16, y=1.05)
            
            for i, (model, title, color) in enumerate(zip(models, titles, colors)):
                conf_values = self.eval_results[model]['confidence']
                correct = np.array(self.eval_results[model]['predictions']) == np.array(self.eval_results[model]['true_labels'])
                
                # Plot correct and incorrect predictions with different colors
                axes[i].hist([np.array(conf_values)[correct], np.array(conf_values)[~correct]], 
                            bins=20, stacked=True, color=[color, 'gray'], 
                            alpha=0.7, label=['Correct', 'Incorrect'])
                
                axes[i].set_title(title, fontsize=14)
                axes[i].set_xlabel('Confidence', fontsize=12)
                axes[i].set_ylabel('Count' if i == 0 else '', fontsize=12)
                axes[i].legend(fontsize=10)
                axes[i].grid(alpha=0.3)
                
            plt.tight_layout()
            
            if save_figures and output_prefix:
                plt.savefig(f"{output_prefix}_confidence.png", dpi=300, bbox_inches='tight')
            
            if show_figures:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"Error in confidence distribution plot: {e}")
        
        # 6. Metacognitive Strategy Evolution
        try:
            strategy_stats = self.metacognitive.get_strategy_stats()
            strategy_counts = {
                'Neural': self.metacognitive.strategy_history.count('neural'),
                'Symbolic': self.metacognitive.strategy_history.count('symbolic'),
                'Hybrid': self.metacognitive.strategy_history.count('hybrid')
            }
            
            # Plot strategy distribution
            plt.figure(figsize=(12, 6))
            wedges, texts, autotexts = plt.pie(
                list(strategy_counts.values()), 
                labels=list(strategy_counts.keys()),
                autopct='%1.1f%%', 
                colors=['#3498db', '#2ecc71', '#9b59b6'],  # Blue, Green, Purple
                startangle=90
            )
            plt.title('Metacognitive Strategy Distribution', fontsize=16)
            
            for autotext in autotexts:
                autotext.set_fontsize(10)
                autotext.set_weight('bold')
                
            plt.tight_layout()
            
            if save_figures and output_prefix:
                plt.savefig(f"{output_prefix}_strategy_dist.png", dpi=300, bbox_inches='tight')
            
            if show_figures:
                plt.show()
            else:
                plt.close()
                
            # Also print the strategy evolution metrics
            print("\nMetacognitive Strategy Evolution:")
            print(f"Neural strategy used: {strategy_counts['Neural']} times ({strategy_counts['Neural']/sum(strategy_counts.values())*100:.1f}%)")
            print(f"Symbolic strategy used: {strategy_counts['Symbolic']} times ({strategy_counts['Symbolic']/sum(strategy_counts.values())*100:.1f}%)")
            print(f"Hybrid strategy used: {strategy_counts['Hybrid']} times ({strategy_counts['Hybrid']/sum(strategy_counts.values())*100:.1f}%)")
            
            if 'correct_neural' in strategy_stats:
                print(f"\nStrategy Effectiveness:")
                print(f"Correct with Neural: {strategy_stats['correct_neural']} cases")
                print(f"Correct with Symbolic: {strategy_stats['correct_symbolic']} cases")
                print(f"Correct with Hybrid: {strategy_stats['correct_hybrid']} cases")
                
        except Exception as e:
            print(f"Error in metacognitive strategy plot: {e}")
        
        # Create summary dictionary
        summary = {
            'neural_accuracy': self.eval_results['neural']['accuracy'],
            'symbolic_accuracy': self.eval_results['symbolic']['accuracy'],
            'nexus_accuracy': self.eval_results['nexus']['accuracy'],
            'agreement_cases': self.eval_results['agreement_cases'],
            'metacognitive_stats': self.metacognitive.get_strategy_stats(),
            'class_f1_scores': {
                'neural': [f1_scores[c, 0] for c in range(self.num_classes)],
                'symbolic': [f1_scores[c, 1] for c in range(self.num_classes)],
                'nexus': [f1_scores[c, 2] for c in range(self.num_classes)]
            }
        }
        
        return summary

    def export_results(self, filename):
        """
        Export detailed evaluation results to CSV
        
        Args:
            filename: Output CSV filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.case_details:
                print("No case details available. Run evaluate() first.")
                return False
                
            # Convert case details to DataFrame
            df = pd.DataFrame(self.case_details)
            
            # Add columns for correctness
            df['neural_correct'] = df['neural_pred'] == df['true_label']
            df['symbolic_correct'] = df['symbolic_pred'] == df['true_label']
            df['nexus_correct'] = df['nexus_pred'] == df['true_label']
            
            # Calculate improvement metrics
            df['nexus_improved'] = ((~df['neural_correct'] | ~df['symbolic_correct']) & df['nexus_correct'])
            
            # Save to CSV
            df.to_csv(filename, index=False)
            print(f"Results exported to {filename}")
            return True
            
        except Exception as e:
            print(f"Error exporting results: {e}")
            return False

# ===========================
# 6. Enhanced Synthetic Medical Dataset
# ===========================
class LargeMedicalDataset:
    """
    Enhanced synthetic medical dataset for large-scale experimentation
    with 10,000 patient cases and realistic symptom patterns
    """
    def __init__(self, num_samples=10000, num_features=20, num_classes=4, imbalance=True, random_state=42):
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_classes = num_classes
        self.imbalance = imbalance
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
        
        # Define symptoms and conditions
        self.init_features_and_classes()
        
        # Generate synthetic data
        self.X, self.y, self.symptom_dict, self.risk_factors = self._generate_data()
        
    def init_features_and_classes(self):
        """Initialize feature names and class definitions"""
        # Extended symptom features
        self.feature_names = [
            # Basic symptoms
            'fever', 'high_fever', 'cough', 'severe_cough', 'fatigue', 
            'shortness_of_breath', 'severe_shortness_of_breath', 'sore_throat',
            'runny_nose', 'body_aches', 'headache', 'chills',
            'loss_of_taste', 'loss_of_smell', 'nausea', 'diarrhea',
            
            # Risk factors (also encoded as features)
            'elderly', 'immunocompromised', 'hypertension', 'diabetes'
        ]
        
        # Disease classes
        self.class_names = ['Common Cold', 'Influenza', 'COVID-19', 'Pneumonia']
        
        # Create symptom patterns for each class (used for data generation)
        self.class_patterns = {
            # Common Cold
            0: {
                'primary': ['runny_nose', 'sore_throat'],
                'secondary': ['cough', 'fatigue', 'headache'],
                'rare': ['fever', 'body_aches'],
                'never': ['loss_of_taste', 'loss_of_smell', 'severe_shortness_of_breath', 'high_fever'],
                'risk_impact': 0.2  # Risk factors have lower impact on cold
            },
            
            # Influenza
            1: {
                'primary': ['fever', 'body_aches', 'chills', 'fatigue'],
                'secondary': ['cough', 'headache', 'sore_throat'],
                'rare': ['high_fever', 'nausea', 'diarrhea'],
                'never': ['loss_of_taste', 'loss_of_smell'],
                'risk_impact': 0.5  # Risk factors have medium impact on flu
            },
            
            # COVID-19
            2: {
                'primary': ['fever', 'cough', 'fatigue', 'loss_of_taste', 'loss_of_smell'],
                'secondary': ['shortness_of_breath', 'body_aches', 'headache', 'sore_throat'],
                'rare': ['high_fever', 'severe_cough', 'severe_shortness_of_breath', 'nausea', 'diarrhea'],
                'never': [],
                'risk_impact': 0.8  # Risk factors have high impact on COVID
            },
            
            # Pneumonia
            3: {
                'primary': ['fever', 'cough', 'shortness_of_breath'],
                'secondary': ['high_fever', 'severe_cough', 'severe_shortness_of_breath', 'fatigue'],
                'rare': ['chills', 'nausea'],
                'never': ['runny_nose', 'loss_of_taste', 'loss_of_smell'],
                'risk_impact': 0.9  # Risk factors have highest impact on pneumonia
            }
        }
        
        # Feature indices for easy lookup
        self.feature_indices = {name: i for i, name in enumerate(self.feature_names)}
        
        # Define risk factors and their indices
        self.risk_factor_names = ['elderly', 'immunocompromised', 'hypertension', 'diabetes']
        self.risk_indices = [self.feature_indices[name] for name in self.risk_factor_names]
        
    def _generate_data(self):
        """Generate synthetic medical data with realistic patterns"""
        # Initialize data arrays
        X = np.zeros((self.num_samples, self.num_features), dtype=np.float32)
        
        # Determine class distribution
        if self.imbalance:
            # Realistic imbalanced distribution: more common colds and flu, fewer COVID/pneumonia
            class_probs = [0.45, 0.35, 0.12, 0.08]  # 45% cold, 35% flu, 12% COVID, 8% pneumonia
        else:
            # Balanced distribution
            class_probs = [0.25, 0.25, 0.25, 0.25]
            
        # Assign classes based on distribution
        y = np.random.choice(self.num_classes, size=self.num_samples, p=class_probs)
        
        # Dictionary to store active symptoms for each patient
        symptom_dict = {}
        
        # Dictionary to store risk factors for each patient
        risk_factors = {}
        
        # Generate data for each patient
        for i in range(self.num_samples):
            active_symptoms = []
            patient_risks = []
            class_id = y[i]
            pattern = self.class_patterns[class_id]
            
            # Add primary symptoms (high probability)
            for symptom in pattern['primary']:
                if np.random.random() > 0.1:  # 90% chance
                    X[i, self.feature_indices[symptom]] = 1
                    active_symptoms.append(symptom)
            
            # Add secondary symptoms (medium probability)
            for symptom in pattern['secondary']:
                if np.random.random() > 0.5:  # 50% chance
                    X[i, self.feature_indices[symptom]] = 1
                    active_symptoms.append(symptom)
            
            # Add rare symptoms (low probability)
            for symptom in pattern['rare']:
                if np.random.random() > 0.8:  # 20% chance
                    X[i, self.feature_indices[symptom]] = 1
                    active_symptoms.append(symptom)
            
            # Make sure symptoms in 'never' list are not added
            for symptom in pattern['never']:
                X[i, self.feature_indices[symptom]] = 0
                if symptom in active_symptoms:
                    active_symptoms.remove(symptom)
            
            # Add risk factors (independent of the disease)
            risk_probs = [0.15, 0.08, 0.20, 0.12]  # Elderly, immunocompromised, hypertension, diabetes
            
            for j, (risk_factor, prob) in enumerate(zip(self.risk_factor_names, risk_probs)):
                if np.random.random() < prob:
                    X[i, self.feature_indices[risk_factor]] = 1
                    patient_risks.append(risk_factor)
                    
                    # Risk factors can influence symptom severity based on the disease
                    risk_impact = pattern['risk_impact']
                    
                    # Increase chance of severe symptoms for high-risk patients
                    if 'fever' in active_symptoms and 'high_fever' not in active_symptoms:
                        if np.random.random() < 0.3 * risk_impact:
                            X[i, self.feature_indices['high_fever']] = 1
                            active_symptoms.append('high_fever')
                            
                    if 'cough' in active_symptoms and 'severe_cough' not in active_symptoms:
                        if np.random.random() < 0.3 * risk_impact:
                            X[i, self.feature_indices['severe_cough']] = 1
                            active_symptoms.append('severe_cough')
                            
                    if 'shortness_of_breath' in active_symptoms and 'severe_shortness_of_breath' not in active_symptoms:
                        if np.random.random() < 0.4 * risk_impact:
                            X[i, self.feature_indices['severe_shortness_of_breath']] = 1
                            active_symptoms.append('severe_shortness_of_breath')
            
            # Store active symptoms for this patient
            symptom_dict[i] = active_symptoms
            
            # Store risk factors for this patient
            risk_factors[i] = patient_risks
        
        return X, y, symptom_dict, risk_factors
    
    def get_train_test_split(self, test_size=0.2, validation_size=0.1):
        """Split data into train, validation, and test sets"""
        indices = np.arange(self.num_samples)
        np.random.shuffle(indices)
        
        test_count = int(self.num_samples * test_size)
        val_count = int(self.num_samples * validation_size)
        train_count = self.num_samples - test_count - val_count
        
        train_indices = indices[:train_count]
        val_indices = indices[train_count:train_count+val_count]
        test_indices = indices[train_count+val_count:]
        
        # Create symptom dictionaries for each split
        train_symptoms = {i: self.symptom_dict[idx] for i, idx in enumerate(train_indices)}
        val_symptoms = {i: self.symptom_dict[idx] for i, idx in enumerate(val_indices)}
        test_symptoms = {i: self.symptom_dict[idx] for i, idx in enumerate(test_indices)}
        
        # Create risk factor dictionaries for each split
        train_risks = {i: self.risk_factors[idx] for i, idx in enumerate(train_indices)}
        val_risks = {i: self.risk_factors[idx] for i, idx in enumerate(val_indices)}
        test_risks = {i: self.risk_factors[idx] for i, idx in enumerate(test_indices)}
        
        return {
            'train': {
                'indices': train_indices,
                'X': self.X[train_indices],
                'y': self.y[train_indices],
                'symptoms': train_symptoms,
                'risks': train_risks
            },
            'val': {
                'indices': val_indices,
                'X': self.X[val_indices],
                'y': self.y[val_indices],
                'symptoms': val_symptoms,
                'risks': val_risks
            },
            'test': {
                'indices': test_indices,
                'X': self.X[test_indices],
                'y': self.y[test_indices],
                'symptoms': test_symptoms,
                'risks': test_risks
            }
        }
    
    def get_dataloader(self, batch_size=32, split_data=None):
        """
        Create data loaders for training, validation, and testing
        
        Args:
            batch_size: Batch size for data loaders
            split_data: Data split dictionary from get_train_test_split()
            
        Returns:
            Dictionary of data loaders
        """ 
        if split_data is None:
            split_data = self.get_train_test_split()
            
        train_tensor_x = torch.tensor(split_data['train']['X'], dtype=torch.float32)
        train_tensor_y = torch.tensor(split_data['train']['y'], dtype=torch.long)
        train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        val_tensor_x = torch.tensor(split_data['val']['X'], dtype=torch.float32)
        val_tensor_y = torch.tensor(split_data['val']['y'], dtype=torch.long)
        val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=1, 
            shuffle=False
        )
        
        test_tensor_x = torch.tensor(split_data['test']['X'], dtype=torch.float32)
        test_tensor_y = torch.tensor(split_data['test']['y'], dtype=torch.long)
        test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=1,  # Batch size of 1 for individual evaluation
            shuffle=False
        )
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'split_data': split_data
        }
    
    def get_class_distribution(self):
        """Get the distribution of classes in the dataset"""
        counts = np.bincount(self.y)
        percentages = 100 * counts / len(self.y)
        distribution = {
            'counts': counts,
            'percentages': percentages,
            'class_names': self.class_names
        }
        return distribution
    
    def visualize_data_distribution(self):
        """Visualize the class distribution and feature frequencies"""
        # Class distribution
        class_dist = self.get_class_distribution()
        counts = class_dist['counts']
        percentages = class_dist['percentages']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart of class distribution
        ax1.bar(self.class_names, counts, color='skyblue')
        ax1.set_ylabel('Number of Patients', fontsize=12)
        ax1.set_title('Class Distribution', fontsize=14)
        
        for i, v in enumerate(counts):
            ax1.text(i, v + 50, f"{v}", ha='center')
            ax1.text(i, v/2, f"{percentages[i]:.1f}%", ha='center', color='white', fontweight='bold')
        
        # Pie chart of class distribution
        wedges, texts, autotexts = ax2.pie(
            counts, 
            labels=self.class_names, 
            autopct='%1.1f%%', 
            startangle=90, 
            colors=['skyblue', 'lightgreen', 'salmon', 'mediumpurple']
        )
        ax2.set_title('Class Distribution (Percentage)', fontsize=14)
        
        plt.tight_layout()
        plt.show()
        
        # Feature (symptom) frequencies
        feature_freq = np.sum(self.X, axis=0) / self.num_samples
        
        plt.figure(figsize=(16, 8))
        bars = plt.bar(self.feature_names, feature_freq * 100, color='lightblue')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Frequency (%)', fontsize=12)
        plt.title('Symptom Frequencies in the Dataset', fontsize=14)
        plt.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5, f"{height:.1f}%", 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # Heatmap of symptom correlations with classes
        plt.figure(figsize=(14, 10))
        symptom_class_corr = np.zeros((len(self.feature_names), len(self.class_names)))
        
        for class_id in range(len(self.class_names)):
            class_mask = self.y == class_id
            for feature_id in range(len(self.feature_names)):
                symptom_class_corr[feature_id, class_id] = np.mean(self.X[class_mask, feature_id])
        
        sns.heatmap(symptom_class_corr, cmap='YlGnBu', 
                   xticklabels=self.class_names, yticklabels=self.feature_names,
                   annot=True, fmt='.2f')
        plt.title('Symptom Prevalence by Disease Class', fontsize=14)
        plt.tight_layout()
        plt.show()

# ===========================
# 7. Main Experiment Function
# ===========================
def run_nexus_experiment(
    num_samples=10000,
    num_features=20,
    num_classes=4,
    imbalance=True,
    batch_size=128,
    num_epochs=10,
    learning_rate=0.001,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    output_dir='results',
    random_state=42
):
    """
    Run a complete experiment with the NEXUS architecture
    
    Args:
        num_samples: Number of synthetic patients to generate
        num_features: Number of symptom features
        num_classes: Number of disease classes
        imbalance: Whether to use realistic class imbalance
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to run on ('cuda' or 'cpu')
        output_dir: Directory to save results
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with experiment results
    """
    print(f"NEXUS Experiment with {num_samples} patients")
    print(f"Running on device: {device}")
    
    # Set random seeds
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    random.seed(random_state)
    
    # Create output directory if it doesn't exist
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Generate synthetic medical data
    print("\n1. Generating synthetic medical data...")
    dataset = LargeMedicalDataset(
        num_samples=num_samples,
        num_features=num_features,
        num_classes=num_classes,
        imbalance=imbalance,
        random_state=random_state
    )
    
    # Visualize data distribution
    print("\nData Distribution:")
    class_dist = dataset.get_class_distribution()
    for i, name in enumerate(dataset.class_names):
        print(f"  {name}: {class_dist['counts'][i]} patients ({class_dist['percentages'][i]:.1f}%)")
    
    # 2. Split data and create data loaders
    print("\n2. Preparing data loaders...")
    data_loaders = dataset.get_dataloader(batch_size=batch_size)
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    test_loader = data_loaders['test']
    split_data = data_loaders['split_data']
    
    print(f"  Training set: {len(split_data['train']['X'])} patients")
    print(f"  Validation set: {len(split_data['val']['X'])} patients")
    print(f"  Test set: {len(split_data['test']['X'])} patients")
    
    # 3. Create and initialize NEXUS model
    print("\n3. Creating NEXUS model...")
    symbol_names = dataset.feature_names
    class_names = dataset.class_names
    
    model = EnhancedNEXUSModel(
        input_dim=num_features,
        num_classes=num_classes,
        num_symbols=len(symbol_names),
        symbol_names=symbol_names,
        class_names=class_names,
        device=device
    )
    
    # 4. Initialize knowledge graph
    print("\n4. Initializing knowledge graph with medical knowledge...")
    kg = model.init_knowledge_graph()
    
    # Add symptom-disease relationships
    for class_id, pattern in dataset.class_patterns.items():
        # Primary symptoms strongly indicate the disease
        for symptom in pattern['primary']:
            if symptom in model.symbol_to_id:
                kg.add_relation(
                    model.symbol_to_id[symptom], 
                    "indicates", 
                    class_id, 
                    weight=0.85
                )
        
        # Secondary symptoms moderately indicate the disease
        for symptom in pattern['secondary']:
            if symptom in model.symbol_to_id:
                kg.add_relation(
                    model.symbol_to_id[symptom], 
                    "indicates", 
                    class_id, 
                    weight=0.6
                )
    
    # Add specific diagnostic rules
    # Common Cold rules
    kg.add_rule([model.symbol_to_id['runny_nose'], model.symbol_to_id['sore_throat']], 0, 0.85)
    
    # Flu rules
    kg.add_rule([model.symbol_to_id['fever'], model.symbol_to_id['body_aches']], 1, 0.8)
    kg.add_rule([model.symbol_to_id['fever'], model.symbol_to_id['chills'], model.symbol_to_id['fatigue']], 1, 0.9)
    
    # COVID-19 rules
    kg.add_rule([model.symbol_to_id['loss_of_taste']], 2, 0.85)
    kg.add_rule([model.symbol_to_id['loss_of_smell']], 2, 0.85)
    kg.add_rule([model.symbol_to_id['fever'], model.symbol_to_id['cough'], model.symbol_to_id['fatigue']], 2, 0.75)
    
    # Pneumonia rules
    kg.add_rule([model.symbol_to_id['fever'], model.symbol_to_id['severe_shortness_of_breath']], 3, 0.85)
    kg.add_rule([model.symbol_to_id['high_fever'], model.symbol_to_id['shortness_of_breath']], 3, 0.9)
    
    # Rules for risk factors
    for risk_factor in ['elderly', 'immunocompromised', 'hypertension', 'diabetes']:
        if risk_factor in model.symbol_to_id:
            # Risk factors increase likelihood of severe diseases (COVID, pneumonia)
            kg.add_relation(
                model.symbol_to_id[risk_factor], 
                "increases_risk", 
                2,  # COVID-19
                weight=0.7
            )
            kg.add_relation(
                model.symbol_to_id[risk_factor], 
                "increases_risk", 
                3,  # Pneumonia
                weight=0.8
            )
    
    # Set symbol-to-class mapping for neural-symbolic interface
    symbol_to_class_dict = {}
    for symbol, symbol_id in model.symbol_to_id.items():
        weights = {}
        
        # Map symptoms to classes based on their relationships
        for class_id, pattern in dataset.class_patterns.items():
            if symbol in pattern['primary']:
                weights[class_id] = 0.85
            elif symbol in pattern['secondary']:
                weights[class_id] = 0.6
            elif symbol in pattern['rare']:
                weights[class_id] = 0.3
            elif symbol in pattern['never']:
                weights[class_id] = -0.5
        
        # Add mapping if any weights were assigned
        if weights:
            symbol_to_class_dict[symbol_id] = weights
    
    model.interface.set_symbol_to_class_mapping(symbol_to_class_dict)
    
    # 5. Train neural component
    print("\n5. Training neural component...")
    train_stats = model.train_neural(
        train_loader, 
        num_epochs=num_epochs, 
        lr=learning_rate,
        scheduler='cosine'
    )
    
    # 6. Evaluate on validation set
    print("\n6. Evaluating on validation set...")
    val_results = model.evaluate(
        val_loader, 
        symptom_dict=split_data['val']['symptoms'],
        feedback=True  # Use validation results to tune metacognitive controller
    )
    
    print(f"\nValidation Results:")
    print(f"  Neural accuracy: {val_results['neural']['accuracy']*100:.2f}%")
    print(f"  Symbolic accuracy: {val_results['symbolic']['accuracy']*100:.2f}%")
    print(f"  NEXUS accuracy: {val_results['nexus']['accuracy']*100:.2f}%")
    
    # 7. Evaluate on test set
    print("\n7. Evaluating on test set...")
    test_results = model.evaluate(
        test_loader, 
        symptom_dict=split_data['test']['symptoms'],
        feedback=False  # No feedback on test set
    )
    
    print(f"\nTest Results:")
    print(f"  Neural accuracy: {test_results['neural']['accuracy']*100:.2f}%")
    print(f"  Symbolic accuracy: {test_results['symbolic']['accuracy']*100:.2f}%")
    print(f"  NEXUS accuracy: {test_results['nexus']['accuracy']*100:.2f}%")
    
    # 8. Visualize results
    print("\n8. Visualizing results...")
    output_prefix = f"{output_dir}/nexus_{num_samples}"
    summary = model.visualize_results(
        output_prefix=output_prefix,
        save_figures=True
    )
    
    # 9. Export detailed results
    print("\n9. Exporting detailed results...")
    model.export_results(f"{output_dir}/nexus_{num_samples}_case_details.csv")
    
    # 10. Analyze specific test cases
    print("\n10. Analyzing specific test cases...")
    case_indices = []
    
    # Find one case for each class where NEXUS outperformed both neural and symbolic
    found_classes = set()
    for i, detail in enumerate(model.case_details):
        if (detail['nexus_pred'] == detail['true_label'] and
            (detail['neural_pred'] != detail['true_label'] or 
             detail['symbolic_pred'] != detail['true_label']) and
            detail['true_label'] not in found_classes and
            len(found_classes) < num_classes):
            
            case_indices.append(i)
            found_classes.add(detail['true_label'])
    
    # Add a few more interesting cases
    # Case where neural and symbolic disagree
    for i, detail in enumerate(model.case_details):
        if detail['neural_pred'] != detail['symbolic_pred'] and i not in case_indices:
            case_indices.append(i)
            break
    
    # Case with multiple risk factors
    for i, detail in enumerate(model.case_details):
        idx = detail['index']
        risks = split_data['test']['risks'].get(idx, [])
        if len(risks) >= 2 and i not in case_indices:
            case_indices.append(i)
            break
    
    # Analyze selected cases
    for i in case_indices:
        detail = model.case_details[i]
        idx = detail['index']
        
        # Get input data and symptoms
        x = torch.tensor(split_data['test']['X'][idx], dtype=torch.float32).unsqueeze(0)
        active_symptoms = split_data['test']['symptoms'].get(idx, [])
        risks = split_data['test']['risks'].get(idx, [])
        
        # Get complete diagnosis with explanations
        result = model.diagnose(x, active_symptoms)
        
        print(f"\nCase {i+1} Analysis:")
        print(f"True diagnosis: {detail['true_class']}")
        print(f"Symptoms: {', '.join(active_symptoms)}")
        
        if risks:
            print(f"Risk factors: {', '.join(risks)}")
        
        print("\nDetailed Diagnosis:")
        print(model.explain_diagnosis(result, detail_level='high'))
        
        print("\nComparison:")
        print(f"Neural: {result['neural']['class_name']} (Confidence: {result['neural']['confidence']:.2f})")
        print(f"Symbolic: {result['symbolic']['class_name']} (Confidence: {result['symbolic']['confidence']:.2f})")
        print(f"NEXUS: {result['nexus']['class_name']} (Confidence: {result['nexus']['confidence']:.2f})")
        print(f"Strategy: {result['nexus']['strategy']['strategy']}")
        print(f"Reason: {result['nexus']['strategy']['explanation']}")
        print("-" * 80)
    
    # Return complete experiment results
    experiment_results = {
        'model': model,
        'dataset': dataset,
        'train_stats': train_stats,
        'val_results': val_results,
        'test_results': test_results,
        'summary': summary,
        'split_data': split_data,
        'config': {
            'num_samples': num_samples,
            'num_features': num_features,
            'num_classes': num_classes,
            'imbalance': imbalance,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'device': device,
            'random_state': random_state
        }
    }
    
    return experiment_results

# ===========================
# 8. Main Module
# ===========================
if __name__ == "__main__":
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="Run NEXUS experiment")
    parser.add_argument("--samples", type=int, default=10000, help="Number of synthetic patient cases")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--no_imbalance", action="store_true", help="Use balanced class distribution")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run on ('cuda' or 'cpu')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"NEXUS Transformer for Medical Diagnosis")
    print(f"Analyzing {args.samples} synthetic patient cases")
    print("=" * 80)
    
    start_time = time.time()
    
    experiment_results = run_nexus_experiment(
        num_samples=args.samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        imbalance=not args.no_imbalance,
        output_dir=args.output,
        device=args.device,
        random_state=args.seed
    )
    
    end_time = time.time()
    
    print("\n" + "=" * 80)
    print(f"Experiment completed in {(end_time - start_time) / 60:.2f} minutes")
    print("=" * 80)
    
    # Final comparative summary
    model = experiment_results['model']
    test_results = experiment_results['test_results']
    
    print("\nFinal Comparative Summary:")
    print("-" * 40)
    print(f"Neural Model Accuracy: {test_results['neural']['accuracy']*100:.2f}%")
    print(f"Symbolic Model Accuracy: {test_results['symbolic']['accuracy']*100:.2f}%")
    print(f"NEXUS Model Accuracy: {test_results['nexus']['accuracy']*100:.2f}%")
    
    # Calculate improvements
    neural_acc = test_results['neural']['accuracy']
    symbolic_acc = test_results['symbolic']['accuracy']
    nexus_acc = test_results['nexus']['accuracy']
    
    best_component = max(neural_acc, symbolic_acc)
    improvement = (nexus_acc - best_component) * 100
    
    print(f"\nNEXUS improvement over best component: {improvement:.2f}%")
    
    # Agreement analysis
    agreement = test_results['agreement_cases']
    total = sum(agreement.values())
    
    print("\nAgreement Analysis:")
    print(f"All models correct: {agreement['all_correct']} cases ({100*agreement['all_correct']/total:.1f}%)")
    print(f"Neural only correct: {agreement['neural_only']} cases ({100*agreement['neural_only']/total:.1f}%)")
    print(f"Symbolic only correct: {agreement['symbolic_only']} cases ({100*agreement['symbolic_only']/total:.1f}%)")
    print(f"NEXUS better than components: {agreement['nexus_better']} cases ({100*agreement['nexus_better']/total:.1f}%)")
    print(f"All models wrong: {agreement['all_wrong']} cases ({100*agreement['all_wrong']/total:.1f}%)")
    
    # Strategy usage
    strategy_stats = model.metacognitive.get_strategy_stats()
    
    print("\nMetacognitive Strategy Usage:")
    if 'neural' in strategy_stats:
        print(f"Neural strategy: {strategy_stats['neural']*100:.1f}%")
        print(f"Symbolic strategy: {strategy_stats['symbolic']*100:.1f}%")
        print(f"Hybrid strategy: {strategy_stats['hybrid']*100:.1f}%")
    
    print("\nConclusion:")
    if nexus_acc > best_component:
        print("NEXUS successfully improved over both neural and symbolic components!")
    elif nexus_acc == best_component:
        print("NEXUS performed equally to the best component.")
    else:
        print("NEXUS did not improve over the best component in this experiment.")
        
    print("\nComplete results saved to:", args.output)
    print("=" * 80)
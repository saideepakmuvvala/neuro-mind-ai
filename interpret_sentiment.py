from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients, visualization as viz
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import seaborn as sns

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model.eval()

# Input
text = "I absolutely loved the film, it was inspiring and beautiful!"
inputs = tokenizer(text, return_tensors="pt")

# Get embeddings from model's embedding layer
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
embeddings = model.distilbert.embeddings(input_ids)

def forward_func(inputs_embeds, attention_mask):
    return model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits

# Predict
outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
pred_class = torch.argmax(outputs.logits, dim=1).item()

# Attribution using Integrated Gradients
ig = IntegratedGradients(forward_func)
attributions, delta = ig.attribute(
    inputs=embeddings,
    additional_forward_args=(attention_mask,),
    target=pred_class,
    return_convergence_delta=True
)

# Normalize attributions across sequence
attributions_sum = attributions.sum(dim=-1).squeeze(0)  # shape: [seq_len]
attributions_sum = attributions_sum / torch.norm(attributions_sum)

# Collapse embedding dimensions into a single attribution score per token
token_importances = attributions.sum(dim=-1).squeeze(0)  # shape: [num_tokens]

# Normalize for visualization (optional)
token_importances = token_importances / torch.norm(token_importances)

# Decode tokens
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# Show token-level importance
for token, score in zip(tokens, token_importances):
    print(f"{token:>12} : {score.item():.4f}")

# Collapse 768 embedding dimensions per token → 1 score per token
token_importances = attributions.sum(dim=-1).squeeze(0)  # shape: [num_tokens]
token_importances = token_importances / torch.norm(token_importances)

# Decode tokens
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# Convert scores to numpy for matplotlib
scores = token_importances.detach().numpy()

# Normalize scores between 0 and 1
scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

# Define color map (red = more important)
cmap = LinearSegmentedColormap.from_list("importance", ["#ffffff", "#ff9999"])

# Display tokens with color highlighting
def visualize_tokens(tokens, scores):
    plt.figure(figsize=(len(tokens), 1.5))
    for i, (token, score) in enumerate(zip(tokens, scores)):
        plt.text(i, 0, token.replace("▁", ""), fontsize=14,
                 backgroundcolor=cmap(score))
    plt.axis('off')
    plt.tight_layout()
    plt.show()

visualize_tokens(tokens, scores)

# Aggregate attribution scores across embedding dimensions
word_attributions = attributions.sum(dim=-1).squeeze(0)  # [seq_len]
word_attributions = word_attributions / torch.norm(word_attributions)  # normalize

# Get predicted class index (use argmax if model output is logits)
preds = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
pred_class_idx = torch.argmax(preds.logits, dim=1).item()

# Create visualization records
vis_data_records = [
    viz.VisualizationDataRecord(
        word_attributions.tolist(),              # Full list of attribution scores per token
        pred_class_idx,                          # Predicted class
        torch.max(preds.logits).item(),          # Confidence
        str(pred_class_idx),                     # True label (mocked)
        str(pred_class_idx),                     # Class label name
        word_attributions.sum().item(),          # Total attribution score
        tokens,                                  # Full token list
        delta.item()                             # Convergence delta
    )
]

viz.visualize_text(vis_data_records)

# Enable attention output
model.config.output_attentions = True

# Run forward again to get attention scores
outputs = model(input_ids=input_ids, attention_mask=attention_mask)

# Extract attention from last layer (or loop over all layers)
attention = outputs.attentions[-1]  # shape: (batch_size, num_heads, seq_len, seq_len)
attentions = outputs.attentions
# Get attention for first head (just to visualize easily)
attn_head = attention[0, 0].detach().numpy()  # shape: (seq_len, seq_len)
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# Plot attention heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(attn_head, xticklabels=tokens, yticklabels=tokens, cmap='viridis', square=True)
plt.title("Attention Map - Last Layer, Head 0")
plt.xlabel("Attended To")
plt.ylabel("Attending From")
plt.tight_layout()
plt.show()

# Print results
print("Predicted class:", pred_class)
print("Attributions shape:", attributions.shape)
print(f"Number of layers: {len(attentions)}")
print(f"Shape of attention from last layer: {attentions[-1].shape}")
print(f"Tokens: {tokens}")
print(f"Attn head shape: {attn_head.shape}")
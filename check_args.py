from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch"
)

print("✅ TrainingArguments accepted!")

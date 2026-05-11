"""SFT baseline training script using Unsloth + TRL."""
import wandb
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

from grpo_pbe.data_generator import load_dataset

# --- Config (same model/LoRA as GRPO for fair comparison) ---
MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LENGTH = 1024
LORA_R = 16
LORA_ALPHA = 16
OUTPUT_DIR = "checkpoints/sft-pbe-1.5b"
WANDB_PROJECT = "grpo-pbe"

SYSTEM_PROMPT = (
    "You solve data transformation tasks. Given input/output examples, "
    "write a Python expression using `x` as the input variable. "
    "Respond with reasoning in <think>...</think> tags, "
    "then the expression in <code>...</code> tags."
)


def make_conversations(train_data: list[dict]) -> list[list[dict]]:
    """Convert training examples to chat conversations with gold responses.

    SFT trains on prompt + gold response pairs. The gold response is just
    <code>gold_code</code> (no <think> trace — SFT only learns the answer).
    """
    conversations = []
    for ex in train_data:
        convo = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["prompt"]},
            {"role": "assistant", "content": f"<code>{ex['gold_code']}</code>"},
        ]
        conversations.append(convo)
    return conversations


def main():
    train_data = load_dataset("data/train.jsonl")
    print(f"Loaded {len(train_data)} training examples")

    # Build conversations and format with chat template
    conversations = make_conversations(train_data)

    # Load model with Unsloth (same config as GRPO)
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
    )

    # Format conversations using chat template
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in conversations
    ]
    hf_dataset = Dataset.from_dict({"text": texts})

    # Training config
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        optim="adamw_8bit",
        logging_steps=10,
        save_steps=200,
        report_to="wandb",
        bf16=True,
        seed=42,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    wandb.init(project=WANDB_PROJECT, name="sft-baseline", config={
        "model": MODEL_NAME,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lr": training_args.learning_rate,
        "epochs": training_args.num_train_epochs,
    })

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=hf_dataset,
    )
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")
    wandb.finish()


if __name__ == "__main__":
    main()

"""
Main runner for single-label classification with Weave tracking.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import argparse
from dotenv import load_dotenv
import random
from typing import List
load_dotenv()

import weave

from evaluation.multinomial_predictor import MultinomialPredictor
from models.llm_multinomial_model import LLMMultinomialModel
from prompts.multinomial_prompt import MultinomialPromptTemplate
from utils.multinomial_datareader import AbstractRosarioDataset
from models.openai_llm import OpenAILLM
from models.huggingface_llm import HuggingFaceLLM

WEAVE_PROJECT = "scibeto-benchmark-evaluation"


def get_llm(model_name: str, provider: str):
    """Get LLM based on provider."""
    if provider == "openai":
        return OpenAILLM(model_name=model_name)
    else:
        return HuggingFaceLLM(model_name=model_name, load_in_4bit=True)


def get_few_shot_examples(dataset, n_examples: int, labels: List[str]) -> List[dict]:
    """
    Get balanced few-shot examples from train set.
    """
    if dataset.train is None:
        raise ValueError("Train split required for few-shot examples")
    
    examples = []
    examples_per_label = max(1, n_examples // len(labels))
    
    for label in labels:
        label_df = dataset.train[dataset.train[dataset.label_column] == label]
        if len(label_df) > 0:
            sampled = label_df.sample(n=min(examples_per_label, len(label_df)), random_state=42)
            for _, row in sampled.iterrows():
                examples.append({
                    "text": row[dataset.input_column],
                    "label": row[dataset.label_column]
                })
    
    random.seed(42)
    random.shuffle(examples)
    
    return examples[:n_examples]


def main():
    parser = argparse.ArgumentParser(description="Run single-label classification")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model name")
    parser.add_argument("--provider", type=str, default="openai",
                        choices=["openai", "huggingface"],
                        help="Model provider")
    parser.add_argument("--data-dir", type=str, default="data/abstracts_rosario",
                        help="Path to dataset directory")
    parser.add_argument("--output-dir", type=str, default="results/multinomial",
                        help="Path to save results")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "dev", "test"])
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples for testing")
    parser.add_argument("--few-shot", type=int, default=0,
                        help="Number of few-shot examples (0 = zero-shot)")
    parser.add_argument("--format", type=str, default="json",
                        choices=["json", "parquet"])
    
    args = parser.parse_args()
    
    weave.init(WEAVE_PROJECT)
    
    print("=" * 60)
    print("SINGLE-LABEL CLASSIFICATION - ABSTRACT ROSARIO")
    print("=" * 60)
    
    # Load dataset
    print("\n[1/4] Loading dataset...")
    dataset = AbstractRosarioDataset(data_dir=args.data_dir)
    print(dataset)
    
    # Create LLM
    print(f"\n[2/4] Loading LLM ({args.provider})...")
    llm = get_llm(args.model, args.provider)
    print(llm)
    
    # Determine shot type
    shot_type = f"few_shot_{args.few_shot}" if args.few_shot > 0 else "zero_shot"
    
    # Get few-shot examples if needed
    few_shot_examples = None
    if args.few_shot > 0:
        print(f"\n[2.5/4] Getting {args.few_shot} few-shot examples...")
        few_shot_examples = get_few_shot_examples(dataset, args.few_shot, dataset.labels)
        print(f"  Got {len(few_shot_examples)} examples")
    
    # Create prompt template
    print("\n[3/4] Creating prompt template...")
    prompt_template = MultinomialPromptTemplate(
        available_labels=dataset.labels,
        language="es",
        examples=few_shot_examples
    )
    print(prompt_template)
    
    # Create model
    model = LLMMultinomialModel(
        llm=llm,
        available_labels=dataset.labels,
        prompt_template=prompt_template
    )
    print(model)
    
    # Create predictor and save results
    print("\n[4/4] Generating predictions...")
    predictor = MultinomialPredictor(model, dataset)
    
    output_subdir = f"{args.output_dir}/{args.model.replace('/', '_')}/{shot_type}"
    predictor.save_predictions(
        split=args.split,
        output_dir=output_subdir,
        max_samples=args.max_samples,
        format=args.format,
        shot_type=shot_type
    )
    
    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
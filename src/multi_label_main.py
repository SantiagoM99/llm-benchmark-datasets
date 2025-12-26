import sys
from pathlib import Path
import argparse
import torch
import json

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from evaluation.multilabel_predictor import MultiLabelPredictor
from evaluation.metrics import compute_multilabel_metrics, save_metrics
from utils.multilabel_datareader import MultiLabelDataset
from utils.jel_categories import get_jel_names
from models.huggingface_llm import HuggingFaceLLM
from models.llm_multilabel_model import LLMMultiLabelModel
from prompts.multilabel_prompt import MultiLabelPromptTemplate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-label classification with LLMs"
    )
    
    # Model arguments
    parser.add_argument(
        "--provider",
        type=str,
        default="huggingface",
        choices=["huggingface", "openai", "anthropic"],
        help="LLM provider to use"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path (e.g., 'Qwen/Qwen2.5-7B-Instruct')"
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit quantization"
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit quantization"
    )
    
    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing the multilabel dataset"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["dev", "test"],
        help="Dataset splits to process (default: dev test)"
    )
    
    # Prompt arguments
    parser.add_argument(
        "--language",
        type=str,
        default="es",
        choices=["es", "en"],
        help="Language for prompts and labels"
    )
    parser.add_argument(
        "--few-shot",
        type=int,
        default=0,
        help="Number of few-shot examples to include"
    )
    
    # Processing arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process per split (for testing)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/multilabel",
        help="Directory to save results"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for this experiment (default: use model name)"
    )
    
    return parser.parse_args()


def get_model_short_name(model_name: str) -> str:
    """Extract a short name from the full model path."""
    return model_name.split("/")[-1]


def main():
    args = parse_args()
    
    print("="*60)
    print("MULTILABEL CLASSIFICATION WITH LLMS")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Data directory: {args.data_dir}")
    print(f"Language: {args.language}")
    print(f"Batch size: {args.batch_size}")
    if args.max_samples:
        print(f"Max samples per split: {args.max_samples}")
    print("="*60)
    
    # Load dataset
    print("\n[1/4] Loading dataset...")
    dataset = MultiLabelDataset(data_dir=args.data_dir)
    print(dataset)
    print(f"Available labels: {dataset.labels}")
    
    # Create LLM
    print("\n[2/4] Loading LLM...")
    if args.provider == "huggingface":
        # Determine dtype
        if torch.cuda.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
            
        llm = HuggingFaceLLM(
            model_name=args.model,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
    else:
        raise NotImplementedError(f"Provider {args.provider} not implemented yet")
    
    # Create prompt template
    print("\n[3/4] Creating prompt template...")
    label_names = get_jel_names(language=args.language)
    prompt_template = MultiLabelPromptTemplate(
        available_labels=dataset.labels,
        language=args.language,
        label_descriptions=label_names,
        num_examples=args.few_shot
    )
    
    # Create multi-label model
    model = LLMMultiLabelModel(
        llm=llm,
        available_labels=dataset.labels,
        prompt_template=prompt_template,
        batch_size=args.batch_size
    )
    print(model)
    
    # Determine experiment name and output directory
    if args.experiment_name:
        exp_name = args.experiment_name
    else:
        exp_name = get_model_short_name(args.model)
        if args.load_in_4bit:
            exp_name += "-4bit"
        elif args.load_in_8bit:
            exp_name += "-8bit"
    
    # Determine shot strategy subdirectory
    if args.few_shot > 0:
        shot_strategy = f"few-shot-{args.few_shot}"
    else:
        shot_strategy = "zero-shot"
    
    # Build complete output path: results/multilabel/model_name/strategy/
    output_base = Path(args.output_dir) / exp_name / shot_strategy
    
    print("\n[4/4] Generating predictions and computing metrics...")
    print(f"Strategy: {shot_strategy}")
    print(f"Output directory: {output_base}")
    
    predictor = MultiLabelPredictor(model, dataset)

    # Load few-shot examples if requested
    if args.few_shot > 0:
        print(f"\nLoading {args.few_shot} few-shot examples from training set...")
        try:
            predictor.load_few_shot_examples(split="train", max_examples=args.few_shot)
        except Exception as e:
            print(f"Warning: Could not load few-shot examples: {e}")
            print("Continuing without few-shot examples...")
    
    # Save experiment metadata
    metadata = {
        "model": args.model,
        "data_dir": args.data_dir,
        "strategy": shot_strategy,
        "few_shot_examples": args.few_shot,
        "batch_size": args.batch_size,
        "language": args.language,
        "quantization": "4bit" if args.load_in_4bit else "8bit" if args.load_in_8bit else "none",
        "max_samples": args.max_samples,
        "splits_processed": args.splits,
    }
    
    metadata_path = output_base / "experiment_config.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Experiment config saved to: {metadata_path}")

    for split in args.splits:
        print(f"\n{'='*60}")
        print(f"Processing split: {split} | Strategy: {shot_strategy}")
        print('='*60)
        
        # Predict
        results_df = predictor.predict_split(
            split=split, 
            batch_size=args.batch_size,
            max_samples=args.max_samples
        )

        # Save predictions
        output_pred = output_base / f"{split}_predictions.parquet"
        output_pred.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_parquet(output_pred, index=False)
        print(f"✓ Predictions saved: {output_pred}")

        # Compute and save metrics
        metrics = compute_multilabel_metrics(
            true_labels=results_df["true_labels"].tolist(),
            pred_labels=results_df["predicted_labels"].tolist(),
            all_labels=dataset.labels,
        )
        output_metrics = output_base / f"{split}_metrics.json"
        save_metrics(metrics, str(output_metrics))
        print(f"✓ Metrics saved: {output_metrics}")
        
        # Print summary metrics
        print(f"\nMetrics Summary for {split}:")
        print(f"  Exact Match Ratio: {metrics.get('exact_match_ratio', 0):.4f}")
        print(f"  Hamming Loss: {metrics.get('hamming_loss', 0):.4f}")
        print(f"  Macro F1: {metrics.get('f1_macro', 0):.4f}")
        print(f"  Micro F1: {metrics.get('f1_micro', 0):.4f}")
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETED")
    print(f"Strategy: {shot_strategy}")
    print(f"Results saved in: {output_base}")
    print("="*60)


if __name__ == "__main__":
    main()
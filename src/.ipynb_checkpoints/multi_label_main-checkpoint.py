import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))


# test_multilabel_light.py
from evaluation.multilabel_predictor import MultiLabelPredictor
from evaluation.metrics import compute_multilabel_metrics, save_metrics
from utils.multilabel_datareader import MultiLabelDataset
from utils.jel_categories import get_jel_names
from models.huggingface_llm import HuggingFaceLLM
from models.llm_multilabel_model import LLMMultiLabelModel
from prompts.multilabel_prompt import MultiLabelPromptTemplate
import torch


def main():
    print("="*60)
    print("TESTING MULTILABEL CLASSIFICATION FOR JEL GENERAL CATEGORIES")
    print("="*60)
    
    # Load dataset
    print("\n[1/4] Loading dataset...")
    dataset = MultiLabelDataset(data_dir="data/multilabel_banrep")
    print(dataset)
    print(f"Available labels: {dataset.labels}")
    
    # Create LLM
    print("\n[2/4] Loading LLM...")
    llm = HuggingFaceLLM(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        load_in_4bit=False,
        torch_dtype=torch.float32 if not torch.cuda.is_available() else torch.float16
    )
    
    # Create prompt template
    print("\n[3/4] Creating prompt template...")
    label_names = get_jel_names(language="es")
    prompt_template = MultiLabelPromptTemplate(
        available_labels=dataset.labels,
        language="es",
        label_descriptions=label_names,
    )
    
    # Create multi-label model
    model = LLMMultiLabelModel(
        llm=llm,
        available_labels=dataset.labels,
        prompt_template=prompt_template,
        batch_size=4
    )
    print(model)
    
    # Create predictor and save results for all splits
    print("\n[4/4] Generating predictions and computing metrics for splits...")
    predictor = MultiLabelPredictor(model, dataset)

    for split in ["dev", "test"]:
        print(f"\nProcessing split: {split}")
        results_df = predictor.predict_split(split=split, batch_size=4)

        # Save predictions
        output_pred = Path(f"results/multilabel/TinyLlama-1.1B/{split}_predictions.parquet")
        output_pred.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_parquet(output_pred, index=False)
        print(f"Predicciones guardadas en: {output_pred}")

        # Compute and save metrics
        metrics = compute_multilabel_metrics(
            true_labels=results_df["true_labels"].tolist(),
            pred_labels=results_df["predicted_labels"].tolist(),
            all_labels=dataset.labels,
        )
        output_metrics = Path(f"results/multilabel/TinyLlama-1.1B/{split}_metrics.json")
        save_metrics(metrics, str(output_metrics))
        print(f"Métricas guardadas en: {output_metrics}")
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()
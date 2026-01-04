import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))


# test_multilabel_light.py
from evaluation.multilabel_predictor import MultiLabelPredictor
from utils.multilabel_datareader import MultiLabelDataset
from models.huggingface_llm import HuggingFaceLLM
from models.llm_multilabel_model import LLMMultiLabelModel
from prompts.multilabel_prompt import MultiLabelPromptTemplate
import torch


def main():
    print("="*60)
    print("TESTING MULTILABEL CLASSIFICATION WITH QWEN2.5-0.5B")
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
    prompt_template = MultiLabelPromptTemplate(
        available_labels=dataset.labels,
        language="es"
    )
    
    # Create multi-label model
    model = LLMMultiLabelModel(
        llm=llm,
        available_labels=dataset.labels,
        prompt_template=prompt_template,
        batch_size=2  # Small batch for testing
    )
    print(model)
    
    # Test with a few examples
    print("\n[4/4] Testing predictions...")
    texts_test, labels_test = dataset.get_texts_and_labels("test")
    sample_texts = texts_test[:5]  # Test with 5 samples
    sample_labels = labels_test[:5]
    
    print(f"\nPredicting {len(sample_texts)} samples...")
    predictions = model.predict(sample_texts)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    # Create multi-label model
    print("\n[4/5] Creating classification model...")
    model = LLMMultiLabelModel(
        llm=llm,
        available_labels=dataset.labels,
        prompt_template=prompt_template,
        batch_size=4
    )
    print(model)
    
    # Create predictor and save results
    print("\n[5/5] Generating and saving predictions...")
    predictor = MultiLabelPredictor(model, dataset)

    predictor.save_predictions(
        split="test",
        output_path=f"results/multilabel/TinyLlama-1.1B/test_predictions.parquet",
        batch_size=4
    )
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()
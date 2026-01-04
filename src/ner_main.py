"""NER main runner with kNN retrieval and self-verification."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import random
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

import weave

from evaluation.ner_predictor import NERPredictor
from models.llm_ner_model import LLMNERModel
from prompts.ner_prompt import NERPromptTemplate
from utils.ner_datareader import EconIEDataset

WEAVE_PROJECT = "scibeto-benchmark-evaluation"


def get_llm(model_name: str, provider: str):
    if provider == "openai":
        from models.openai_llm import OpenAILLM
        return OpenAILLM(model_name=model_name)
    elif provider == "azure":
        from models.openai_llm import OpenAILLM
        return OpenAILLM(model_name=model_name, use_azure=True)
    elif provider == "huggingface":
        from models.huggingface_llm import HuggingFaceLLM
        return HuggingFaceLLM(model_name=model_name, load_in_4bit=True)
    raise ValueError(f"Provider '{provider}' not supported")


def bio_to_entities(tokens: List[str], tags: List[str]) -> List[Dict[str, str]]:
    """Convert BIO tags to entity list."""
    entities = []
    current_entity = None
    current_tokens = []
    
    for token, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            if current_entity:
                entities.append({"text": " ".join(current_tokens), "type": current_entity})
            current_entity = tag[2:]
            current_tokens = [token]
        elif tag.startswith("I-") and current_entity:
            current_tokens.append(token)
        else:
            if current_entity:
                entities.append({"text": " ".join(current_tokens), "type": current_entity})
                current_entity = None
                current_tokens = []
    
    if current_entity:
        entities.append({"text": " ".join(current_tokens), "type": current_entity})
    
    return entities


def get_train_examples(dataset: EconIEDataset) -> List[Dict]:
    """Get all training examples with entities."""
    sentences = dataset.get_text_sentences("train")
    tokens_list, tags_list = dataset.get_sentences_and_labels("train")
    
    examples = []
    for sent, tokens, tags in zip(sentences, tokens_list, tags_list):
        entities = bio_to_entities(tokens, tags)
        if entities:
            examples.append({"text": sent, "entities": entities})
    return examples


def get_few_shot_examples_random(examples: List[Dict], n: int) -> List[Dict]:
    """Random few-shot selection."""
    random.seed(42)
    if len(examples) > n:
        return random.sample(examples, n)
    return examples[:n]


def get_few_shot_examples_knn(
    examples: List[Dict],
    query: str,
    n: int,
    retriever,
    entity_types: List[str],
    balanced: bool = True
) -> List[Dict]:
    """kNN-based few-shot selection."""
    if balanced:
        return retriever.retrieve_balanced(query, n, entity_types)
    return retriever.retrieve(query, n)


def main():
    parser = argparse.ArgumentParser(description="NER with kNN and verification")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--provider", type=str, default="openai",
                        choices=["openai", "azure", "huggingface"])
    parser.add_argument("--data-dir", type=str, default="data/ner_econ_ie")
    parser.add_argument("--output-dir", type=str, default="results/ner")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--few-shot", type=int, default=0)
    # GPT-NER improvements
    parser.add_argument("--use-knn", action="store_true", help="Use kNN retrieval for few-shot")
    parser.add_argument("--use-verification", action="store_true", help="Use self-verification")
    parser.add_argument("--balanced-knn", action="store_true", help="Balance kNN by entity type")
    
    args = parser.parse_args()
    weave.init(WEAVE_PROJECT)
    
    print("=" * 60)
    print("NER - ECON-IE")
    print(f"kNN: {args.use_knn} | Verification: {args.use_verification}")
    print("=" * 60)
    
    # Load dataset
    print("\n[1/4] Loading dataset...")
    dataset = EconIEDataset(data_dir=args.data_dir)
    print(dataset)
    
    # Load LLM
    print(f"\n[2/4] Loading LLM ({args.provider}: {args.model})...")
    llm = get_llm(args.model, args.provider)
    
    # Setup kNN retriever if needed
    retriever = None
    train_examples = None
    if args.few_shot > 0:
        train_examples = get_train_examples(dataset)
        print(f"  Loaded {len(train_examples)} training examples")
        
        if args.use_knn:
            print("  Building kNN index...")
            from utils.knn_retriever import KNNRetriever
            retriever = KNNRetriever()
            retriever.build_index(train_examples)
    
    # Setup verification if needed
    verify_fn = None
    if args.use_verification:
        from utils.self_verification import verify_entities
        verify_fn = lambda sent, ents: verify_entities(sent, ents, llm)
    
    # Determine output path
    mode_parts = []
    if args.few_shot > 0:
        mode_parts.append(f"few_shot_{args.few_shot}")
        if args.use_knn:
            mode_parts.append("knn")
    else:
        mode_parts.append("zero_shot")
    if args.use_verification:
        mode_parts.append("verified")
    shot_type = "_".join(mode_parts)
    
    # Get base few-shot examples (for non-kNN mode)
    few_shot_examples = None
    if args.few_shot > 0 and not args.use_knn:
        few_shot_examples = get_few_shot_examples_random(train_examples, args.few_shot)
        print(f"  Selected {len(few_shot_examples)} random examples")
    
    # Create prompt template
    print("\n[3/4] Creating prompt template...")
    prompt_template = NERPromptTemplate(
        entity_types=dataset.entity_types,
        language="es",
        examples=few_shot_examples
    )
    
    # Create model with optional kNN and verification
    model = LLMNERModel(
        llm=llm,
        entity_types=dataset.entity_types,
        prompt_template=prompt_template
    )
    
    # If using kNN, we need custom prediction loop
    if args.use_knn and args.few_shot > 0:
        print("\n[4/4] Generating predictions with kNN retrieval...")
        results = run_with_knn(
            dataset=dataset,
            model=model,
            retriever=retriever,
            n_few_shot=args.few_shot,
            entity_types=dataset.entity_types,
            balanced=args.balanced_knn,
            verify_fn=verify_fn,
            split=args.split,
            max_samples=args.max_samples
        )
        save_results(results, args.output_dir, args.model, shot_type)
    else:
        # Standard prediction (with optional verification)
        print("\n[4/4] Generating predictions...")
        if args.use_verification:
            run_with_verification(
                dataset=dataset,
                model=model,
                verify_fn=verify_fn,
                output_dir=args.output_dir,
                model_name=args.model,
                shot_type=shot_type,
                split=args.split,
                max_samples=args.max_samples
            )
        else:
            predictor = NERPredictor(model, dataset)
            output_subdir = f"{args.output_dir}/{args.model.replace('/', '_')}/{shot_type}"
            predictor.save_predictions(
                split=args.split,
                output_dir=output_subdir,
                max_samples=args.max_samples,
                shot_type=shot_type
            )
    
    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)


def run_with_knn(
    dataset: EconIEDataset,
    model: LLMNERModel,
    retriever,
    n_few_shot: int,
    entity_types: List[str],
    balanced: bool,
    verify_fn,
    split: str,
    max_samples: int
) -> List[Dict]:
    """Run predictions with per-sentence kNN retrieval."""
    from tqdm import tqdm
    
    sentences = dataset.get_text_sentences(split)
    tokens_list, tags_list = dataset.get_sentences_and_labels(split)
    
    if max_samples:
        sentences = sentences[:max_samples]
        tokens_list = tokens_list[:max_samples]
        tags_list = tags_list[:max_samples]
    
    results = []
    total_filtered = 0
    
    for sent, tokens, tags in tqdm(zip(sentences, tokens_list, tags_list), total=len(sentences)):
        expected = bio_to_entities(tokens, tags)
        
        # Get kNN examples for this sentence
        knn_examples = get_few_shot_examples_knn(
            retriever.examples, sent, n_few_shot, retriever, entity_types, balanced
        )
        
        # Create dynamic prompt with kNN examples
        dynamic_prompt = NERPromptTemplate(
            entity_types=entity_types,
            language="es",
            examples=knn_examples
        )
        
        # Predict
        prompt = dynamic_prompt.create_prompt(sent)
        response = model.llm.generate(prompt=prompt, max_tokens=512, temperature=0.0)
        predicted = dynamic_prompt.parse_response(response)
        
        # Optional verification
        if verify_fn:
            before = len(predicted)
            predicted = verify_fn(sent, predicted)
            total_filtered += before - len(predicted)
        
        results.append({
            "sentence": sent,
            "expected_entities": expected,
            "predicted_entities": predicted
        })
    
    if verify_fn:
        print(f"  Filtered {total_filtered} entities via verification")
    
    return results


def run_with_verification(
    dataset: EconIEDataset,
    model: LLMNERModel,
    verify_fn,
    output_dir: str,
    model_name: str,
    shot_type: str,
    split: str,
    max_samples: int
):
    """Run predictions with verification only (no kNN)."""
    from tqdm import tqdm
    import json
    from pathlib import Path
    
    sentences = dataset.get_text_sentences(split)
    tokens_list, tags_list = dataset.get_sentences_and_labels(split)
    
    if max_samples:
        sentences = sentences[:max_samples]
        tokens_list = tokens_list[:max_samples]
        tags_list = tags_list[:max_samples]
    
    results = []
    total_filtered = 0
    
    for sent, tokens, tags in tqdm(zip(sentences, tokens_list, tags_list), total=len(sentences)):
        expected = bio_to_entities(tokens, tags)
        predicted = model.predict_single(sent)
        
        before = len(predicted)
        predicted = verify_fn(sent, predicted)
        total_filtered += before - len(predicted)
        
        results.append({
            "sentence": sent,
            "expected_entities": expected,
            "predicted_entities": predicted
        })
    
    print(f"  Filtered {total_filtered} entities via verification")
    save_results(results, output_dir, model_name, shot_type)


def save_results(results: List[Dict], output_dir: str, model_name: str, shot_type: str):
    """Save results to JSON."""
    import json
    from pathlib import Path
    
    output_path = Path(output_dir) / model_name.replace("/", "_") / shot_type
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / "predictions.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)
    
    print(f"  Saved to {output_file}")


if __name__ == "__main__":
    main()
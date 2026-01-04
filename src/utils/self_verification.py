"""Self-verification for NER entities."""
from typing import List, Dict

ENTITY_DEFS = {
    "outcome": "resultados o métricas de estudios",
    "intervention": "programas o políticas evaluadas",
    "population": "grupos de estudio o participantes",
    "effect_size": "tamaños de efecto o porcentajes",
    "coreference": "referencias a entidades previas"
}


def create_verification_prompt(sentence: str, entity_text: str, entity_type: str) -> str:
    """Create verification prompt for a single entity."""
    defn = ENTITY_DEFS.get(entity_type, entity_type)
    return f"""Verifica si la palabra es una entidad del tipo indicado.

Tipo: {entity_type}
Definición: {defn}

Oración: {sentence}
¿Es "{entity_text}" una entidad de tipo {entity_type}?

Responde SOLO "Sí" o "No".
Respuesta:"""


def parse_verification(response: str) -> bool:
    """Parse yes/no response."""
    r = response.strip().lower()
    return r.startswith("sí") or r.startswith("si") or r.startswith("yes")


def verify_entities(
    sentence: str,
    entities: List[Dict[str, str]],
    llm,
    max_tokens: int = 10
) -> List[Dict[str, str]]:
    """Filter entities through self-verification."""
    verified = []
    for ent in entities:
        text, etype = ent.get("text", ""), ent.get("type", "")
        if not text or not etype:
            continue
        
        prompt = create_verification_prompt(sentence, text, etype)
        try:
            response = llm.generate(prompt=prompt, max_tokens=max_tokens, temperature=0.0)
            if parse_verification(response):
                verified.append(ent)
        except Exception:
            verified.append(ent)  # Keep on error
    
    return verified
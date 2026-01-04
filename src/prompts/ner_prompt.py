"""
Prompt template for Named Entity Recognition (NER) tasks.
"""
from typing import List, Optional, Dict
import json
import re
from prompts.base_prompt import BasePromptTemplate


class NERPromptTemplate(BasePromptTemplate):
    """
    Prompt template for NER tasks.
    
    Supports two output formats:
    - 'json': Returns entities as JSON list
    - 'inline': Returns text with entities marked inline
    """
    
    def __init__(
        self,
        entity_types: List[str],
        language: str = "es",
        output_format: str = "json",
        examples: Optional[List[dict]] = None
    ):
        """
        Initialize the NER prompt template.
        
        Args:
            entity_types: List of entity types (e.g., ['outcome', 'intervention', 'population'])
            language: Prompt language ('es' or 'en')
            output_format: 'json' or 'inline'
            examples: Optional few-shot examples [{"text": ..., "entities": [...]}, ...]
        """
        super().__init__(language=language)
        self.entity_types = entity_types
        self.output_format = output_format
        self.examples = examples or []
    
    @property
    def is_few_shot(self) -> bool:
        return len(self.examples) > 0
    
    def _format_examples_json(self) -> str:
        """Format few-shot examples for JSON output."""
        if not self.examples:
            return ""
        
        formatted = []
        for ex in self.examples:
            entities_json = json.dumps(ex['entities'], ensure_ascii=False)
            formatted.append(f"Texto: {ex['text']}\nEntidades: {entities_json}")
        
        return "\n\n".join(formatted)
    
    def _format_examples_inline(self) -> str:
        """Format few-shot examples for inline output."""
        if not self.examples:
            return ""
        
        formatted = []
        for ex in self.examples:
            # Convert entities to inline format
            inline_text = self._entities_to_inline(ex['text'], ex['entities'])
            formatted.append(f"Texto: {ex['text']}\nEntidades: {inline_text}")
        
        return "\n\n".join(formatted)
    
    def _entities_to_inline(self, text: str, entities: List[dict]) -> str:
        """Convert entity list to inline marked text."""
        # Sort entities by start position (reverse to avoid index shifting)
        sorted_entities = sorted(entities, key=lambda x: x.get('start', 0), reverse=True)
        
        result = text
        for ent in sorted_entities:
            entity_text = ent.get('text', '')
            entity_type = ent.get('type', '')
            # Mark entity in text
            marked = f"[{entity_text}]({entity_type})"
            if 'start' in ent and 'end' in ent:
                result = result[:ent['start']] + marked + result[ent['end']:]
            else:
                result = result.replace(entity_text, marked, 1)
        
        return result
    
    def create_prompt(self, text: str, **kwargs) -> str:
        """
        Create a prompt for NER.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Formatted prompt
        """
        entity_types_str = ", ".join(self.entity_types)
        
        # Few-shot section
        few_shot_section = ""
        if self.is_few_shot:
            if self.output_format == "json":
                few_shot_section = f"\n\nEjemplos:\n\n{self._format_examples_json()}\n\n---"
            else:
                few_shot_section = f"\n\nEjemplos:\n\n{self._format_examples_inline()}\n\n---"
        
        # Entity type definitions for the prompt
        entity_definitions = """
Definiciones de tipos de entidades:
- outcome: Resultados, métricas, variables de resultado del estudio (ej: "ingresos", "productividad", "mortalidad")
- intervention: Tratamientos, programas, políticas evaluadas (ej: "microcréditos", "programa de transferencias", "vacunación")
- population: Grupos de estudio, participantes, muestras (ej: "hogares rurales", "niños", "empresas pequeñas")
- effect_size: Tamaños de efecto, magnitudes, porcentajes de cambio (ej: "aumentó 15%", "0.5 desviaciones estándar")
- coreference: Palabras que REFIEREN a entidades mencionadas previamente - pronombres o sustantivos que hacen referencia anafórica (ej: "el programa" refiriéndose a una intervención anterior, "estos resultados", "ambas intervenciones", "los grupos")
"""
        
        if self.language == "es":
            if self.output_format == "json":
                prompt = f"""Eres un experto en extracción de entidades nombradas de textos económicos y científicos.

Tipos de entidades a extraer: {entity_types_str}
{entity_definitions}
Instrucciones:
- Extrae TODAS las entidades del texto, incluyendo correferencias
- Las correferencias son palabras que REFIEREN a otras entidades mencionadas en otro contexto
- Responde ÚNICAMENTE con una lista JSON
- Cada entidad debe tener: "text" (texto exacto), "type" (tipo de entidad)
- Si no hay entidades, responde con una lista vacía: []
- No agregues explicaciones{few_shot_section}

Texto:
{text}

Entidades (JSON):"""
            else:
                prompt = f"""Eres un experto en extracción de entidades nombradas de textos económicos y científicos.

Tipos de entidades a extraer: {entity_types_str}
{entity_definitions}
Instrucciones:
- Marca las entidades en el texto usando el formato [entidad](tipo)
- Ejemplo: [los participantes](population) recibieron [el tratamiento](intervention)
- Las correferencias son palabras que refieren a entidades mencionadas en otro contexto
- Si no hay entidades, repite el texto sin cambios
- No agregues explicaciones{few_shot_section}

Texto:
{text}

Texto con entidades marcadas:"""
        else:
            # English entity definitions
            entity_definitions_en = """
Entity type definitions:
- outcome: Results, metrics, outcome variables of the study (e.g., "income", "productivity", "mortality")
- intervention: Treatments, programs, policies being evaluated (e.g., "microcredit", "transfer program", "vaccination")
- population: Study groups, participants, samples (e.g., "rural households", "children", "small businesses")
- effect_size: Effect sizes, magnitudes, percentage changes (e.g., "increased 15%", "0.5 standard deviations")
- coreference: Words that REFER to entities mentioned previously - pronouns or nouns making anaphoric reference (e.g., "the program" referring to a prior intervention, "these results", "both interventions", "the groups")
"""
            if self.output_format == "json":
                prompt = f"""You are an expert in named entity extraction from economic and scientific texts.

Entity types to extract: {entity_types_str}
{entity_definitions_en}
Instructions:
- Extract ALL entities from the text, including coreferences
- Coreferences are words that REFER to other entities mentioned in another context
- Respond ONLY with a JSON list
- Each entity must have: "text" (exact text), "type" (entity type)
- If no entities, respond with an empty list: []
- Do not add explanations{few_shot_section}

Text:
{text}

Entities (JSON):"""
            else:
                prompt = f"""You are an expert in named entity extraction from economic and scientific texts.

Entity types to extract: {entity_types_str}
{entity_definitions_en}
Instructions:
- Mark entities in the text using the format [entity](type)
- Example: [the participants](population) received [the treatment](intervention)
- Coreferences are words that refer to entities mentioned in another context
- If no entities, repeat the text unchanged
- Do not add explanations{few_shot_section}

Text:
{text}

Text with marked entities:"""
        
        return prompt
    
    def parse_response(self, response: str) -> List[Dict[str, str]]:
        """
        Parse the model's response to extract entities.
        
        Args:
            response: Raw model response
            
        Returns:
            List of entities [{"text": ..., "type": ...}, ...]
        """
        response = response.strip()
        
        if self.output_format == "json":
            return self._parse_json_response(response)
        else:
            return self._parse_inline_response(response)
    
    def _parse_json_response(self, response: str) -> List[Dict[str, str]]:
        """Parse JSON format response."""
        # Try to find JSON array in response
        try:
            # Find JSON array pattern
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                entities = json.loads(match.group())
                # Validate and clean entities
                valid_entities = []
                for ent in entities:
                    if isinstance(ent, dict) and 'text' in ent and 'type' in ent:
                        # Skip if text or type is None
                        if ent['text'] is None or ent['type'] is None:
                            continue
                        # Normalize entity type
                        ent_type = str(ent['type']).lower().replace(' ', '_')
                        if ent_type in self.entity_types:
                            valid_entities.append({
                                'text': str(ent['text']),
                                'type': ent_type
                            })
                return valid_entities
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
        
        return []
    
    def _parse_inline_response(self, response: str) -> List[Dict[str, str]]:
        """Parse inline format response."""
        entities = []
        
        # Pattern: [entity text](entity_type)
        pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        matches = re.findall(pattern, response)
        
        for text, entity_type in matches:
            ent_type = entity_type.lower().replace(' ', '_')
            if ent_type in self.entity_types:
                entities.append({
                    'text': text,
                    'type': ent_type
                })
        
        return entities
    
    def __repr__(self) -> str:
        return (
            f"NERPromptTemplate(\n"
            f"  entity_types={self.entity_types},\n"
            f"  language='{self.language}',\n"
            f"  output_format='{self.output_format}',\n"
            f"  few_shot={self.is_few_shot} ({len(self.examples)} examples)\n"
            f")"
        )
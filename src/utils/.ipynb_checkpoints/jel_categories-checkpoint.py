"""
JEL general category definitions for multi-label classification prompts.

Provides mappings from single-letter JEL codes to human-readable names
in English and Spanish. Use these to enrich prompts and improve LLM
classification performance.
"""

from typing import Dict


# English names for JEL general categories (A–Z)
JEL_GENERAL_CATEGORIES_EN: Dict[str, str] = {
    "A": "General Economics and Teaching",
    "B": "History of Economic Thought, Methodology, and Heterodox Approaches",
    "C": "Mathematical and Quantitative Methods",
    "D": "Microeconomics",
    "E": "Macroeconomics and Monetary Economics",
    "F": "International Economics",
    "G": "Financial Economics",
    "H": "Public Economics",
    "I": "Health, Education, and Welfare",
    "J": "Labor and Demographic Economics",
    "K": "Law and Economics",
    "L": "Industrial Organization",
    "M": "Business Administration and Business Economics • Marketing • Accounting • Personnel Economics",
    "N": "Economic History",
    "O": "Economic Development, Innovation, Technological Change, and Growth",
    "P": "Political Economy and Comparative Economic Systems",
    "Q": "Agricultural and Natural Resource Economics • Environmental and Ecological Economics",
    "R": "Urban, Rural, Regional, Real Estate, and Transportation Economics",
    "Y": "Miscellaneous Categories",
    "Z": "Other Special Topics",
}


# Spanish names for JEL general categories (A–Z)
JEL_GENERAL_CATEGORIES_ES: Dict[str, str] = {
    "A": "Economía General y Enseñanza",
    "B": "Historia del Pensamiento Económico, Metodología y Enfoques Heterodoxos",
    "C": "Métodos Matemáticos y Cuantitativos",
    "D": "Microeconomía",
    "E": "Macroeconomía y Economía Monetaria",
    "F": "Economía Internacional",
    "G": "Economía Financiera",
    "H": "Economía Pública",
    "I": "Salud, Educación y Bienestar",
    "J": "Economía Laboral y Demográfica",
    "K": "Derecho y Economía",
    "L": "Organización Industrial",
    "M": "Administración y Economía de la Empresa • Marketing • Contabilidad • Economía del Personal",
    "N": "Historia Económica",
    "O": "Desarrollo Económico, Innovación, Cambio Tecnológico y Crecimiento",
    "P": "Economía Política y Sistemas Económicos Comparados",
    "Q": "Economía Agrícola y de Recursos Naturales • Economía Ambiental y Ecológica",
    "R": "Economía Urbana, Rural, Regional, Inmobiliaria y de Transporte",
    "Y": "Categorías Misceláneas",
    "Z": "Otros Temas Especiales",
}


def get_jel_names(language: str = "es") -> Dict[str, str]:
    """Return JEL general category names for the given language.

    Args:
        language: "es" for Spanish, "en" for English

    Returns:
        Dict mapping single-letter JEL code to human-readable name.
    """
    if language.lower().startswith("es"):
        return JEL_GENERAL_CATEGORIES_ES
    return JEL_GENERAL_CATEGORIES_EN

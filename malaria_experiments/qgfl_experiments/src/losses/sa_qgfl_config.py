"""
Species-Adaptive QGFL Configuration (D3 Dataset)

This configuration file defines the parameters for SA-QGFL (Species-Adaptive
Quality-Guided Focal Loss) for multi-species Plasmodium detection on the D3 dataset.

Three novel components are configured here:
1. MCP (Morphological Complexity Priors) - Per-species gamma values
2. PWA (Prevalence-Weighted Alpha) - Per-species alpha values
3. HQS (Hierarchical Quality Scoring) - Two-tier quality weights

Based on:
- D3 dataset distribution (Guemas et al. 2024 / Zenodo DOI: 10.5281/zenodo.8358829)
- Parasitology literature for morphological complexity
- WHO mortality data for clinical importance weighting
"""

from typing import Dict

# =============================================================================
# CLASS CONFIGURATION
# =============================================================================

# Number of classes in D3 species detection
NC: int = 5

# Class names for D3 (5 classes)
CLASS_NAMES: Dict[int, str] = {
    0: 'Uninfected',
    1: 'P_falciparum',
    2: 'P_ovale',
    3: 'P_malariae',
    4: 'P_vivax',
}

# Class IDs for easy reference
CLASS_UNINFECTED = 0
CLASS_P_FALCIPARUM = 1
CLASS_P_OVALE = 2
CLASS_P_MALARIAE = 3
CLASS_P_VIVAX = 4

# Rare/minority species (for stratified analysis)
RARE_SPECIES = ['P_ovale', 'P_malariae', 'P_vivax']

# =============================================================================
# COMPONENT 1: MCP (Morphological Complexity Priors)
# =============================================================================
# Based on parasitology: harder species to detect = higher gamma value
# Higher gamma increases focus on difficult samples
#
# Rationale:
# - Uninfected (γ=2.0): Background cells, easy to classify
# - P. falciparum (γ=5.0): Distinctive ring forms and banana gametocytes
# - P. ovale (γ=7.0): Subtle fimbriated RBCs, harder to distinguish
# - P. malariae (γ=9.0): Rarest species, band forms and "basket" schizonts
# - P. vivax (γ=7.0): Enlarged RBCs with Schuffner's dots, variable morphology

MCP_GAMMA: Dict[int, float] = {
    0: 2.0,   # Uninfected - background (easy)
    1: 5.0,   # P. falciparum - distinctive features (moderate)
    2: 7.0,   # P. ovale - subtle features (hard)
    3: 9.0,   # P. malariae - rarest, hardest to detect
    4: 7.0,   # P. vivax - variable morphology (hard)
}

# =============================================================================
# COMPONENT 2: PWA (Prevalence-Weighted Alpha)
# =============================================================================
# Based on D3 distribution (97.2% uninfected, 2.8% parasites)
# Combined with clinical mortality weights from WHO data
#
# D3 Training Set Distribution:
# - Class 0 (Uninfected): 1,579,752 (97.2%)
# - Class 1 (P. falciparum): 36,241 (2.2%)
# - Class 2 (P. ovale): 3,771 (0.23%)
# - Class 3 (P. malariae): 2,163 (0.13%) - RAREST
# - Class 4 (P. vivax): 3,246 (0.20%)
#
# Alpha weighting strategy:
# - Low alpha for majority class (suppress gradient contribution)
# - High alpha for rare/minority classes (boost gradient contribution)
# - P. malariae gets highest alpha (rarest + moderate mortality)
# - P. falciparum boosted due to highest mortality

PWA_ALPHA: Dict[int, float] = {
    0: 0.01,  # Uninfected - suppress heavily (97.2% of data)
    1: 0.25,  # P. falciparum (2.2%) - boost, highest mortality
    2: 0.20,  # P. ovale (0.23%) - boost, rare
    3: 0.30,  # P. malariae (0.13%) - boost most, rarest species
    4: 0.24,  # P. vivax (0.20%) - boost, rare + relapsing concern
}

# Clinical importance weights (WHO mortality data)
# Used in PWA computation - can be adjusted for different clinical priorities
CLINICAL_WEIGHTS: Dict[int, float] = {
    0: 0.1,   # Uninfected - no clinical concern
    1: 1.5,   # P. falciparum - highest mortality (cerebral malaria)
    2: 0.8,   # P. ovale - moderate, can cause relapses
    3: 0.7,   # P. malariae - lowest mortality but chronic
    4: 1.0,   # P. vivax - significant (relapses, growing resistance)
}

# =============================================================================
# COMPONENT 3: HQS (Hierarchical Quality Scoring)
# =============================================================================
# Two-tier quality assessment:
# 1. Infection detection: Is it infected at all? (binary question)
# 2. Species discrimination: Which species is it? (multi-class question)
#
# Clinical rationale:
# - First priority: Correctly identify infected vs uninfected
# - Second priority: Correctly identify the species (for treatment)
# - Species discrimination weighted higher because correct species ID
#   is crucial for appropriate treatment selection

HQS_INFECTION_WEIGHT: float = 0.3   # Weight for infection detection quality
HQS_SPECIES_WEIGHT: float = 0.7     # Weight for species discrimination quality

# =============================================================================
# DEFAULT QGFL PARAMETERS (can be overridden)
# =============================================================================

DEFAULT_DIFFICULTY_THRESHOLD: float = 0.6
DEFAULT_QUALITY_MARGIN: float = 0.3

# =============================================================================
# EVALUATION PARAMETERS (Guemas et al. 2024 compatible)
# =============================================================================

GUEMAS_CONF_THRESHOLD: float = 0.25
GUEMAS_IOU_THRESHOLD: float = 0.45
GUEMAS_AGNOSTIC_NMS: bool = True

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_class_alpha(class_id: int) -> float:
    """Get PWA alpha value for a given class ID."""
    return PWA_ALPHA.get(class_id, 0.5)

def get_class_gamma(class_id: int) -> float:
    """Get MCP gamma value for a given class ID."""
    return MCP_GAMMA.get(class_id, 2.0)

def get_class_name(class_id: int) -> str:
    """Get class name for a given class ID."""
    return CLASS_NAMES.get(class_id, f'Unknown_{class_id}')

def is_parasite_class(class_id: int) -> bool:
    """Check if class ID represents a parasite (not uninfected)."""
    return class_id > 0 and class_id < NC

def get_parasite_class_ids() -> list:
    """Get list of parasite class IDs (excluding uninfected)."""
    return list(range(1, NC))

def get_rare_species_ids() -> list:
    """Get list of rare species class IDs (P. ovale, P. malariae, P. vivax)."""
    return [CLASS_P_OVALE, CLASS_P_MALARIAE, CLASS_P_VIVAX]

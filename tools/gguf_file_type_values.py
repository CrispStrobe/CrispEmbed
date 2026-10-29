"""Standard GGUF Q4 header metadata; actual tensor storage is checked separately."""
import operator
Q4_K_S=14
LEGACY_INTERNAL_Q4_K=12


def q4_k_file_type(value):
    if isinstance(value,bool):return False
    try:return operator.index(value)in(LEGACY_INTERNAL_Q4_K,Q4_K_S)
    except TypeError:return False

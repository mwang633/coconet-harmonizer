"""
Coconet: 4-Part Harmony Generator

A PyTorch implementation of Coconet for generating Bach-style
4-part (SATB) harmonizations from a melody.
"""

from .model import Coconet
from .data import BachChoraleDataset, load_bach_chorales
from .harmonize import Harmonizer

__all__ = ["Coconet", "BachChoraleDataset", "load_bach_chorales", "Harmonizer"]

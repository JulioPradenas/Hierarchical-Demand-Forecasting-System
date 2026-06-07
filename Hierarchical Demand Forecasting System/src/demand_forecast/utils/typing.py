from typing import Literal

HierarchyLevel = Literal[
    "total", "state", "store", "category", "department", "item_store"
]

HIERARCHY_LEVELS: list[HierarchyLevel] = [
    "total",
    "state",
    "store",
    "category",
    "department",
    "item_store",
]

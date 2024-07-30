from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset
from typing import Dict, List

def create_similarity_tree(root_synsets: List[Synset], degree: int) -> Dict[str, any]:
    """
    Creates a tree structure of synsets starting from given root synsets, expanding to a specified degree.
    Each level of the tree is created using hyponyms of the synsets from the previous level, 
    selecting the most similar synsets at each step.

    Args:
        root_synsets (List[Synset]): A list of synsets to be used as the roots of the trees.
        degree (int): The depth of the tree to be created.

    Returns:
        Dict[str, any]: A dictionary representing the tree of synsets.
    """
    
    def similarity(syn1: Synset, syn2: Synset) -> float:
        """
        Calculates the Wu-Palmer similarity between two synsets.

        Args:
            syn1 (Synset): The first synset.
            syn2 (Synset): The second synset.

        Returns:
            float: The similarity score between the two synsets.
        """
        return syn1.wup_similarity(syn2) or 0

    def create_subtree(root: Synset, current_level: int, max_level: int) -> Dict[str, any]:
        """
        Recursively creates a subtree of synsets starting from the given root synset.

        Args:
            root (Synset): The root synset for the current subtree.
            current_level (int): The current depth level of the subtree.
            max_level (int): The maximum depth level for the tree.

        Returns:
            Dict[str, any]: A dictionary representing the subtree of synsets.
        """
        if current_level > max_level:
            return {}
        
        subtree = {}
        hyponyms = root.hyponyms()
        if not hyponyms:
            return subtree

        # Sorts the leaves / nodes in order of compliance with the root
        hyponyms = sorted(hyponyms, key=lambda syn: similarity(root, syn), reverse=True)
        
        # New level is composed by n // 2 nodes. They are most compliant with the root
        num_hyponyms = max(1, len(hyponyms) // 2)
        top_hyponyms = hyponyms[:num_hyponyms]

        for hyponym in top_hyponyms:
            subtree[hyponym.name()] = create_subtree(hyponym, current_level + 1, max_level)

        return subtree

    tree = {}
    for synset in root_synsets:
        tree[synset.name()] = create_subtree(synset, 1, degree)
    
    return tree

#!/usr/bin/env python3
"""
Generate out-of-distribution (OOD) WordNet test dataset by rephrasing questions.
This script takes the original wordnet_test.json and creates wordnet_test_ood1_test.json
with rephrased questions to test model robustness.
"""

import json
import random
import string
from typing import Dict, List, Any
import copy

def get_rephrase_templates() -> List[str]:
    """
    Return a list of alternative phrasings for 'Which of the following is/are types of X?'
    """
    templates = [
        "From the options below, which one(s) represent types of {}?",
        "Among the following choices, which is/are categories of {}?",
        "Select the option(s) that classify as types of {}.",
        "Which option(s) from the list below are varieties of {}?",
        "From these alternatives, which one(s) are forms of {}?",
        "Identify which of the following represents types of {}.",
        "Which of these options can be classified as types of {}?",
        "Among the given choices, which one(s) are kinds of {}?",
        "Select which of the following are subtypes of {}.",
        "Which option(s) below represent categories of {}?",
        "From the choices provided, which are types of {}?",
        "Determine which of the following are varieties of {}.",
        "Which of these alternatives are forms of {}?",
        "Among the options, which one(s) classify as types of {}?",
        "Select the choice(s) that are kinds of {}.",
    ]
    return templates

def extract_concept_from_question(question: str) -> str:
    """
    Extract the concept from the original question format.
    E.g., "Which of the following is/are types of debris?" -> "debris"
    """
    if "Which of the following is/are types of " in question:
        concept = question.replace("Which of the following is/are types of ", "").rstrip("?")
        return concept
    return ""

def rephrase_question(original_question: str, templates: List[str], idx: int) -> str:
    """
    Rephrase the original question using a random template.
    """
    concept = extract_concept_from_question(original_question)
    if not concept:
        return original_question
    
    # Choose a random template
    random.seed(idx)
    template = random.choice(templates)
    return template.format(concept)

def get_special_characters() -> List[str]:
    """
    Return a list of special characters to insert into text.
    """
    special_chars = [
        '!', '@', '#', '$', '%', '&', '*', '^'
    ]
    return special_chars

def insert_special_characters(text: str, insertion_ratio: float, seed: int) -> str:
    """
    Insert special characters into text based on insertion ratio.
    insertion_ratio: 0.25 for 1/4, 0.5 for 2/4, 0.75 for 3/4, 1.0 for 4/4
    """
    random.seed(seed)
    all_special_chars = get_special_characters()
    
    # Select subset of special characters based on ratio
    num_chars_to_use = max(1, int(len(all_special_chars) * insertion_ratio))
    special_chars = all_special_chars[:num_chars_to_use]
    
    # Calculate number of characters to insert
    text_length = len(text)
    num_insertions = int(text_length * insertion_ratio * 0.15)
    
    if num_insertions == 0:
        return text
    
    # Convert text to list for easier manipulation
    text_list = list(text)
    
    # Generate random positions to insert characters
    # Avoid inserting at the very beginning or end
    valid_positions = list(range(1, len(text_list)))
    if len(valid_positions) == 0:
        return text
    
    # Select random positions
    insertion_positions = random.sample(
        valid_positions, 
        min(num_insertions, len(valid_positions))
    )
    
    # Sort positions in descending order to maintain correct indices when inserting
    insertion_positions.sort(reverse=True)
    
    # Insert special characters
    for pos in insertion_positions:
        special_char = random.choice(special_chars)
        text_list.insert(pos, special_char)
    
    return ''.join(text_list)

def generate_ood_dataset(input_file: str, output_file: str, seed: int = 42) -> None:
    """
    Generate OOD dataset by rephrasing questions in the input file.
    """
    random.seed(seed)
    
    # Load the original dataset
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} entries from {input_file}")
    
    # Get rephrase templates
    templates = get_rephrase_templates()
    print(f"Using {len(templates)} different rephrase templates")
    
    # Process each entry
    ood_data = []
    for idx, entry in enumerate(data):
        # Create a copy of the original entry
        ood_entry = copy.deepcopy(entry)
        
        # Rephrase the input question
        original_input = entry['input']
        rephrased_input = rephrase_question(original_input, templates, idx)
        ood_entry['input'] = rephrased_input
        
        # Add metadata about the transformation
        ood_entry['original_input'] = original_input
        ood_entry['transformation'] = 'rephrase_question'
        
        ood_data.append(ood_entry)
    
    # Save the OOD dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ood_data, f, indent=2, ensure_ascii=False)
    
    print(f"Generated OOD dataset with {len(ood_data)} entries")
    print(f"Saved to {output_file}")
    
    # Show some examples
    print("\nExample transformations:")
    for i in range(min(5, len(ood_data))):
        print(f"\nExample {i+1}:")
        print(f"Original: {ood_data[i]['original_input']}")
        print(f"Rephrased: {ood_data[i]['input']}")

def generate_ood_with_special_chars(input_file: str, output_file: str, insertion_ratio: float, seed: int = 42) -> None:
    """
    Generate OOD dataset by adding special characters to rephrased questions.
    """
    random.seed(seed)
    
    # Load the original dataset
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} entries from {input_file}")
    print(f"Using insertion ratio: {insertion_ratio} ({insertion_ratio*100:.0f}% of text length)")
    
    # Get rephrase templates
    templates = get_rephrase_templates()
    
    # Process each entry
    ood_data = []
    for idx, entry in enumerate(data):
        # Create a copy of the original entry
        ood_entry = copy.deepcopy(entry)
        
        # Rephrase the input question first
        original_input = entry['input']
        rephrased_input = rephrase_question(original_input, templates, idx)
        
        # Then insert special characters
        modified_input = insert_special_characters(rephrased_input, insertion_ratio, idx)
        ood_entry['input'] = modified_input
        
        # Add metadata about the transformation
        ood_entry['original_input'] = original_input
        ood_entry['rephrased_input'] = rephrased_input
        ood_entry['transformation'] = f'rephrase_and_special_chars_{insertion_ratio}'
        ood_entry['insertion_ratio'] = insertion_ratio
        
        ood_data.append(ood_entry)
    
    # Save the OOD dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ood_data, f, indent=2, ensure_ascii=False)
    
    print(f"Generated OOD dataset with {len(ood_data)} entries")
    print(f"Saved to {output_file}")
    
    # Show some examples
    print("\nExample transformations:")
    for i in range(min(3, len(ood_data))):
        print(f"\nExample {i+1}:")
        print(f"Original: {ood_data[i]['original_input']}")
        print(f"Rephrased: {ood_data[i]['rephrased_input']}")
        print(f"With special chars: {ood_data[i]['input']}")

def main():
    input_file = "dataset_files/icl/wordnet_test.json"
    base_output_dir = "dataset_files/icl/"

    print("Generating WordNet OOD test datasets...")
    
    # Generate OOD1: Only rephrased questions
    print("\n=== Generating OOD1 (rephrased questions only) ===")
    ood1_file = base_output_dir + "wordnet_test_ood1_test.json"
    generate_ood_dataset(input_file, ood1_file)
    
    # Generate OOD2-5: Rephrased + special characters with increasing ratios
    ratios = {
        2: 0.25,  # 1/4
        3: 0.50,  # 2/4
        4: 0.75,  # 3/4
        5: 1.00   # 4/4
    }
    
    for ood_num, ratio in ratios.items():
        print(f"\n=== Generating OOD{ood_num} (rephrased + {ratio*100:.0f}% special chars) ===")
        output_file = base_output_dir + f"wordnet_test_ood{ood_num}_test.json"
        generate_ood_with_special_chars(input_file, output_file, ratio)
    
    print("\n=== All OOD datasets generated successfully! ===")
    print("Generated files:")
    for i in range(1, 6):
        print(f"  - wordnet_test_ood{i}_test.json")
        
    input_file = "dataset_files/icl/wordnet_train.json"
    base_output_dir = "dataset_files/icl/"

    print("Generating WordNet OOD test datasets...")
    
    # Generate OOD1: Only rephrased questions
    print("\n=== Generating OOD1 (rephrased questions only) ===")
    ood1_file = base_output_dir + "wordnet_test_ood1_train.json"
    generate_ood_dataset(input_file, ood1_file)
    
    # Generate OOD2-5: Rephrased + special characters with increasing ratios
    ratios = {
        2: 0.25,  # 1/4
        3: 0.50,  # 2/4
        4: 0.75,  # 3/4
        5: 1.00   # 4/4
    }
    
    for ood_num, ratio in ratios.items():
        print(f"\n=== Generating OOD{ood_num} (rephrased + {ratio*100:.0f}% special chars) ===")
        output_file = base_output_dir + f"wordnet_test_ood{ood_num}_train.json"
        generate_ood_with_special_chars(input_file, output_file, ratio)
    
    print("\n=== All OOD datasets generated successfully! ===")
    print("Generated files:")
    for i in range(1, 6):
        print(f"  - wordnet_test_ood{i}_train.json")

if __name__ == "__main__":
    main()

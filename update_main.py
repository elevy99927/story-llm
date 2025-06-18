#!/usr/bin/env python3
"""
Script to update the main function in hebrew_story_analyzer.py
"""

import re

def update_main_function():
    # Read the current file
    with open('hebrew_story_analyzer.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the main function
    main_pattern = r'def main\(\):(.*?)if __name__ == "__main__":'
    main_match = re.search(main_pattern, content, re.DOTALL)
    
    if not main_match:
        print("Could not find main function")
        return
    
    # Get the current main function
    current_main = main_match.group(1)
    
    # Add the new arguments
    new_args = """
    parser.add_argument('--improve', action='store_true',
                        help='Suggest improvements for sentences')
    parser.add_argument('--improve-count', type=int, default=5,
                        help='Number of sentences to improve (default: 5)')
    """
    
    # Find where to insert the new arguments
    args_pattern = r'(parser\.add_argument\(\'--detect-ai\'.*?\))'
    args_match = re.search(args_pattern, current_main, re.DOTALL)
    
    if not args_match:
        print("Could not find where to insert new arguments")
        return
    
    # Insert the new arguments
    updated_main = current_main.replace(
        args_match.group(1),
        f"{args_match.group(1)}{new_args}"
    )
    
    # Add the code to call improve_text
    improve_code = """
    # Improve text if requested
    if args.improve:
        improvements = improve_text(text, max_sentences=args.improve_count)
    """
    
    # Find where to insert the improve code
    insert_pattern = r'(# Detect AI-written text if requested.*?ai_lines, ai_info = detect_ai_written_text\(text\))'
    insert_match = re.search(insert_pattern, updated_main, re.DOTALL)
    
    if not insert_match:
        print("Could not find where to insert improve code")
        return
    
    # Insert the improve code
    updated_main = updated_main.replace(
        insert_match.group(1),
        f"{insert_match.group(1)}\n{improve_code}"
    )
    
    # Add the code to print improvements
    print_code = """
    # Print improvement suggestions if available
    if 'improvements' in locals():
        print("\\nהצעות לשיפור ניסוח:")  # "Improvement suggestions"
        if improvements['improved_count'] > 0:
            for i, (original, improved) in enumerate(improvements['improvements'].items(), 1):
                print(f"\\n{i}. מקור: {original}")
                print(f"   שיפור: {improved}")
        else:
            print("לא נמצאו משפטים הדורשים שיפור.")  # "No sentences requiring improvement found"
    """
    
    # Find where to insert the print code
    print_pattern = r'(# Print AI detection results if available.*?)'
    print_match = re.search(print_pattern, updated_main, re.DOTALL)
    
    if not print_match:
        print("Could not find where to insert print code")
        return
    
    # Insert the print code
    updated_main = updated_main.replace(
        print_match.group(1),
        f"{print_match.group(1)}\n{print_code}"
    )
    
    # Update the main function in the file
    updated_content = content.replace(main_match.group(1), updated_main)
    
    # Write the updated file
    with open('hebrew_story_analyzer.py', 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print("Successfully updated main function")

if __name__ == "__main__":
    update_main_function()
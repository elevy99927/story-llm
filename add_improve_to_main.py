#!/usr/bin/env python3
"""
Add the improve functionality to the hebrew_story_analyzer.py main function
"""

import os

def add_improve_to_main():
    # Read the current file
    with open('hebrew_story_analyzer.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the main function
    main_start = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('def main():'):
            main_start = i
            break
    
    if main_start == -1:
        print("Could not find main function")
        return
    
    # Find the argument parser section
    parser_start = -1
    for i in range(main_start, len(lines)):
        if 'parser = argparse.ArgumentParser' in lines[i]:
            parser_start = i
            break
    
    if parser_start == -1:
        print("Could not find argument parser")
        return
    
    # Find where to add the new arguments
    args_pos = -1
    for i in range(parser_start, len(lines)):
        if '--detect-ai' in lines[i]:
            args_pos = i + 1
            break
    
    if args_pos == -1:
        print("Could not find where to add arguments")
        return
    
    # Add the new arguments
    new_args = [
        "    parser.add_argument('--improve', action='store_true',\n",
        "                        help='Suggest improvements for sentences')\n",
        "    parser.add_argument('--improve-count', type=int, default=5,\n",
        "                        help='Number of sentences to improve (default: 5)')\n"
    ]
    
    # Insert the new arguments
    lines = lines[:args_pos] + new_args + lines[args_pos:]
    
    # Find where to add the improve text code
    detect_ai_pos = -1
    for i in range(main_start, len(lines)):
        if 'if args.detect_ai:' in lines[i]:
            detect_ai_pos = i + 2  # Add after the detect_ai block
            break
    
    if detect_ai_pos == -1:
        print("Could not find where to add improve text code")
        return
    
    # Add the improve text code
    improve_code = [
        "\n    # Improve text if requested\n",
        "    if args.improve:\n",
        "        improvements = improve_text(text, max_sentences=args.improve_count)\n"
    ]
    
    # Insert the improve text code
    lines = lines[:detect_ai_pos] + improve_code + lines[detect_ai_pos:]
    
    # Find where to add the print improvements code
    print_pos = -1
    for i in range(main_start, len(lines)):
        if 'if \'ai_info\' in locals():' in lines[i]:
            # Find the end of this block
            for j in range(i, len(lines)):
                if lines[j].strip() and not lines[j].startswith(' '):
                    print_pos = j
                    break
            break
    
    if print_pos == -1:
        # Try another approach - find the last print statement
        for i in range(len(lines) - 1, main_start, -1):
            if 'print(' in lines[i]:
                print_pos = i + 1
                break
    
    if print_pos == -1:
        print("Could not find where to add print improvements code")
        return
    
    # Add the print improvements code
    print_code = [
        "\n    # Print improvement suggestions if available\n",
        "    if 'improvements' in locals():\n",
        "        print(\"\\nהצעות לשיפור ניסוח:\")  # \"Improvement suggestions\"\n",
        "        if improvements['improved_count'] > 0:\n",
        "            for i, (original, improved) in enumerate(improvements['improvements'].items(), 1):\n",
        "                print(f\"\\n{i}. מקור: {original}\")\n",
        "                print(f\"   שיפור: {improved}\")\n",
        "        else:\n",
        "            print(\"לא נמצאו משפטים הדורשים שיפור.\")  # \"No sentences requiring improvement found\"\n"
    ]
    
    # Insert the print improvements code
    lines = lines[:print_pos] + print_code + lines[print_pos:]
    
    # Write the updated file
    with open('hebrew_story_analyzer.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("Successfully added improve functionality to main function")

if __name__ == "__main__":
    add_improve_to_main()
#!/usr/bin/env python3
"""
Hebrew Story Analyzer - A simplified version for analyzing Hebrew text stories
"""

import argparse
import os
import re
import statistics
from typing import Dict, Any, List, Tuple
import time

def load_text(file_path: str) -> str:
    """Load text from a file with proper Hebrew encoding."""
    encodings = ['utf-8', 'cp1255', 'iso-8859-8']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
                print(f"Successfully loaded file with {encoding} encoding")
                return content
        except UnicodeDecodeError:
            continue
    
    print(f"Error: Could not decode file {file_path} with any known Hebrew encoding")
    exit(1)

def analyze_writing_style(text: str) -> Tuple[List[str], Dict[str, Any]]:
    """
    Analyze writing style and identify lines that don't match the main style.
    Returns a list of inconsistent lines and style metrics.
    """
    print("Analyzing writing style...")
    
    # Split text into lines and sentences
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Calculate style metrics for each line
    metrics = []
    for line in lines:
        if len(line) < 3:  # Skip very short lines
            continue
            
        # Calculate basic metrics - use a pattern that works with Hebrew
        words = re.findall(r'[\w\u0590-\u05FF]+', line)  # Include Hebrew Unicode range
        if not words:
            continue
            
        avg_word_len = sum(len(word) for word in words) / len(words) if words else 0
        sentence_len = len(words)
        punctuation_count = len(re.findall(r'[.,;:!?،؛]', line))  # Include Arabic punctuation
        
        metrics.append({
            'line': line,
            'avg_word_len': avg_word_len,
            'sentence_len': sentence_len,
            'punctuation_ratio': punctuation_count / len(line) if len(line) > 0 else 0
        })
    
    if not metrics:
        return [], {'error': 'Not enough text to analyze style'}
    
    # Calculate average metrics to determine the main style
    avg_word_lengths = [m['avg_word_len'] for m in metrics]
    avg_sentence_lengths = [m['sentence_len'] for m in metrics]
    avg_punct_ratios = [m['punctuation_ratio'] for m in metrics]
    
    # Calculate mean and standard deviation
    mean_word_len = statistics.mean(avg_word_lengths)
    stdev_word_len = statistics.stdev(avg_word_lengths) if len(avg_word_lengths) > 1 else 0
    
    mean_sent_len = statistics.mean(avg_sentence_lengths)
    stdev_sent_len = statistics.stdev(avg_sentence_lengths) if len(avg_sentence_lengths) > 1 else 0
    
    mean_punct_ratio = statistics.mean(avg_punct_ratios)
    stdev_punct_ratio = statistics.stdev(avg_punct_ratios) if len(avg_punct_ratios) > 1 else 0
    
    # Identify outliers (lines that don't match the main style)
    outliers = []
    for m in metrics:
        # Check if metrics are outside 2 standard deviations
        is_outlier = False
        
        if stdev_word_len > 0 and abs(m['avg_word_len'] - mean_word_len) > 2 * stdev_word_len:
            is_outlier = True
        
        if stdev_sent_len > 0 and abs(m['sentence_len'] - mean_sent_len) > 2 * stdev_sent_len:
            is_outlier = True
            
        if stdev_punct_ratio > 0 and abs(m['punctuation_ratio'] - mean_punct_ratio) > 2 * stdev_punct_ratio:
            is_outlier = True
            
        if is_outlier:
            outliers.append(m['line'])
    
    # Save outliers to file
    with open('bad_lines.txt', 'w', encoding='utf-8') as f:
        f.write("Lines that don't match the main writing style:\n\n")
        for line in outliers:
            f.write(line + '\n\n')
    
    style_info = {
        'avg_word_length': mean_word_len,
        'avg_sentence_length': mean_sent_len,
        'avg_punctuation_ratio': mean_punct_ratio,
        'inconsistent_lines_count': len(outliers),
        'total_lines_analyzed': len(metrics)
    }
    
    return outliers, style_info

def extract_characters(text: str) -> List[str]:
    """
    Extract character names from the text using simple heuristics.
    """
    print("Extracting character names...")
    
    # Look for common Hebrew name patterns
    # This is a simplified approach - for better results, use NER models
    
    # Find quoted names (often character names in dialogue)
    quoted_names = re.findall(r'"([^"]{2,20})"', text)
    
    # Find capitalized words that might be names
    # In Hebrew, this doesn't work as well as in English, but we can look for words after specific markers
    name_markers = ['פרופסור', 'דוקטור', 'מר', 'גברת', 'גב\'', 'ד"ר']
    marked_names = []
    
    for marker in name_markers:
        pattern = rf'{marker}\s+([א-ת][א-ת\s\'"]{{2,20}})'
        found_names = re.findall(pattern, text)
        marked_names.extend(found_names)
    
    # Combine and clean up the names
    all_names = quoted_names + marked_names
    cleaned_names = []
    
    for name in all_names:
        name = name.strip()
        if len(name) > 2 and name not in cleaned_names:
            cleaned_names.append(name)
    
    return cleaned_names[:10]  # Return top 10 names

def extract_locations(text: str) -> List[str]:
    """
    Extract location names from the text using simple heuristics.
    """
    print("Extracting locations...")
    
    # Common location markers in Hebrew
    location_markers = ['ב', 'ל', 'מ', 'אל', 'על יד', 'ליד', 'בתוך', 'מחוץ ל']
    
    # Find locations after markers
    locations = []
    
    for marker in location_markers:
        pattern = rf'{marker}([א-ת][א-ת\s\'"]{{2,30}})'
        found_locations = re.findall(pattern, text)
        locations.extend(found_locations)
    
    # Clean up locations
    cleaned_locations = []
    for loc in locations:
        loc = loc.strip()
        if len(loc) > 3 and loc not in cleaned_locations:
            cleaned_locations.append(loc)
    
    return cleaned_locations[:10]  # Return top 10 locations

def simple_sentiment_analysis(text: str) -> str:
    """
    Perform very simple sentiment analysis based on keyword matching.
    """
    print("Analyzing sentiment...")
    
    # Define positive and negative word lists in Hebrew
    positive_words = [
        'טוב', 'יפה', 'נהדר', 'מצוין', 'נפלא', 'אהבה', 'שמחה', 'אושר',
        'הצלחה', 'חיוך', 'צחוק', 'אור', 'תקווה', 'חיובי', 'מרגש'
    ]
    
    negative_words = [
        'רע', 'מכוער', 'נורא', 'איום', 'כישלון', 'עצב', 'כעס', 'פחד',
        'דאגה', 'בכי', 'כאב', 'חושך', 'ייאוש', 'שלילי', 'מדכא'
    ]
    
    # Count occurrences
    positive_count = 0
    negative_count = 0
    
    for word in positive_words:
        positive_count += len(re.findall(r'\b' + word + r'\b', text))
    
    for word in negative_words:
        negative_count += len(re.findall(r'\b' + word + r'\b', text))
    
    # Determine overall sentiment
    if positive_count > negative_count:
        return "חיובי (Positive)"
    elif negative_count > positive_count:
        return "שלילי (Negative)"
    else:
        return "ניטרלי (Neutral)"

def extract_main_themes(text: str) -> List[str]:
    """
    Extract potential themes from the text using keyword frequency.
    """
    print("Extracting themes...")
    
    # Define common theme categories in Hebrew
    theme_categories = {
        'אהבה': ['אהבה', 'רומנטיקה', 'זוגיות', 'יחסים', 'לב'],
        'מלחמה': ['מלחמה', 'קרב', 'לחימה', 'חייל', 'נשק', 'צבא'],
        'משפחה': ['משפחה', 'הורים', 'ילדים', 'אבא', 'אמא', 'אח', 'אחות'],
        'מסע': ['מסע', 'הרפתקה', 'דרך', 'נסיעה', 'גילוי'],
        'זהות': ['זהות', 'עצמי', 'מי אני', 'חיפוש עצמי', 'משמעות'],
        'כוח': ['כוח', 'שליטה', 'עוצמה', 'השפעה', 'מנהיגות'],
        'מדע': ['מדע', 'טכנולוגיה', 'מחקר', 'גילוי', 'המצאה'],
        'היסטוריה': ['היסטוריה', 'עבר', 'עתיק', 'מסורת', 'מורשת'],
        'פנטזיה': ['קסם', 'פנטזיה', 'דמיון', 'על-טבעי', 'יצורים'],
        'מתח': ['מתח', 'מסתורין', 'פשע', 'חקירה', 'סוד']
    }
    
    # Count occurrences of theme keywords
    theme_counts = {theme: 0 for theme in theme_categories}
    
    for theme, keywords in theme_categories.items():
        for keyword in keywords:
            count = len(re.findall(r'\b' + keyword + r'\b', text))
            theme_counts[theme] += count
    
    # Get top themes
    top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Return only themes that have at least one keyword match
    return [theme for theme, count in top_themes if count > 0][:5]

def main():
    parser = argparse.ArgumentParser(description='Analyze a Hebrew story text')
    parser.add_argument('file', help='Path to the text file containing the Hebrew story')
    parser.add_argument('--analyze-style', action='store_true',
                        help='Analyze writing style and identify inconsistent lines')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found")
        exit(1)
    
    print(f"Loading text from {args.file}...")
    text = load_text(args.file)
    print(f"Loaded {len(text)} characters ({len(text.split())} words)")
    
    start_time = time.time()
    
    # Analyze writing style if requested
    if args.analyze_style:
        outliers, style_info = analyze_writing_style(text)
    
    # Extract characters
    characters = extract_characters(text)
    
    # Extract locations
    locations = extract_locations(text)
    
    # Simple sentiment analysis
    sentiment = simple_sentiment_analysis(text)
    
    # Extract themes
    themes = extract_main_themes(text)
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*50)
    print(f"Analysis completed in {elapsed_time:.2f} seconds")
    print("="*50)
    
    print("\nדמויות מרכזיות (Main Characters):")
    for char in characters:
        print(f"- {char}")
    
    print("\nמקומות (Locations):")
    for loc in locations:
        print(f"- {loc}")
    
    print(f"\nסנטימנט כללי (Overall Sentiment): {sentiment}")
    
    print("\nנושאים מרכזיים (Main Themes):")
    for theme in themes:
        print(f"- {theme}")
    
    # Print style analysis results if available
    if 'style_info' in locals():
        print("\nניתוח סגנון כתיבה (Writing Style Analysis):")
        print(f"- אורך מילה ממוצע (Average word length): {style_info['avg_word_length']:.2f} characters")
        print(f"- אורך משפט ממוצע (Average sentence length): {style_info['avg_sentence_length']:.2f} words")
        print(f"- שורות לא עקביות (Inconsistent lines): {style_info['inconsistent_lines_count']} out of {style_info['total_lines_analyzed']}")
        print(f"- שורות לא עקביות נשמרו לקובץ (Inconsistent lines saved to) bad_lines.txt")
    
if __name__ == "__main__":
    main()
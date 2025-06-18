from typing import Dict, Any, List, Tuple
import os
import re
import hashlib
import pickle
import argparse
import time

# Define cache settings
USE_CACHE = True

# AI detection thresholds
AI_SENTENCE_LENGTH_MIN = 10
AI_SENTENCE_LENGTH_MAX = 25
AI_LEXICAL_DIVERSITY_THRESHOLD = 0.68
AI_PUNCT_RATIO_MIN = 0.06
AI_PUNCT_RATIO_MAX = 0.14
AI_SCORE_THRESHOLD = 2 #1.8

def get_cache_dir():
    """Get or create cache directory for models."""
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.model_cache')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def improve_sentence(sentence: str, model_name: str = "google/mt5-small") -> str:
    """
    Improve a sentence by suggesting better phrasing.
    
    Args:
        sentence: The original sentence to improve
        model_name: The model to use for improvement
        
    Returns:
        An improved version of the sentence
    """
    try:
        from transformers import MT5ForConditionalGeneration, MT5Tokenizer
        
        print(f"Improving sentence phrasing...")
        
        # Create a hash for caching
        sentence_hash = hashlib.md5(sentence.encode()).hexdigest()
        cache_file = os.path.join(get_cache_dir(), f"improve_{sentence_hash}.pkl")
        
        # Check cache first
        if USE_CACHE and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}. Will recompute.")
        
        # Load model
        tokenizer = MT5Tokenizer.from_pretrained(model_name)
        model = MT5ForConditionalGeneration.from_pretrained(model_name)
        
        # Create prompt
        prompt = f"שפר את הניסוח של המשפט הבא: {sentence}"
        
        # Generate improved sentence
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(
            inputs.input_ids,
            max_length=len(sentence.split()) * 2,  # Allow for longer output
            min_length=len(sentence.split()) // 2,  # But not too short
            num_beams=5,
            temperature=0.7,
            do_sample=True,
            no_repeat_ngram_size=2,
            repetition_penalty=1.3
        )
        
        improved = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the result
        improved = re.sub(r'שפר את הניסוח של המשפט הבא:', '', improved)
        improved = re.sub(r'<extra_id_\d+>', '', improved)  # Remove special tags
        improved = improved.strip()
        
        # If the model didn't produce a good result, return original
        if not improved or len(improved) < len(sentence) / 2:
            return sentence
            
        # Cache the result
        if USE_CACHE:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(improved, f)
            except Exception as e:
                print(f"Warning: Could not save to cache: {e}")
                
        return improved
        
    except Exception as e:
        print(f"Error improving sentence: {e}")
        return sentence  # Return original if there's an error


def improve_text(text: str, max_sentences: int = 5) -> Dict[str, Any]:
    """
    Improve the phrasing of a text by suggesting better wording for selected sentences.
    
    Args:
        text: The text to improve
        max_sentences: Maximum number of sentences to improve
        
    Returns:
        Dictionary with original and improved sentences
    """
    print("Analyzing text for improvement suggestions...")
    
    # Split text into sentences
    sentences = re.split(r'([.!?])\s+', text)
    
    # Recombine sentences with their punctuation
    proper_sentences = []
    i = 0
    while i < len(sentences) - 1:
        if i + 1 < len(sentences) and len(sentences[i+1]) == 1 and sentences[i+1] in '.!?':
            proper_sentences.append(sentences[i] + sentences[i+1])
            i += 2
        else:
            proper_sentences.append(sentences[i])
            i += 1
    
    # Filter out very short sentences
    sentences = [s.strip() for s in proper_sentences if len(s.strip()) > 10]
    
    # Score sentences based on complexity and awkwardness
    scored_sentences = []
    for sentence in sentences:
        score = 0
        
        # Long sentences might need improvement
        if len(sentence.split()) > 20:
            score += 1
            
        # Sentences with many commas might be complex
        if sentence.count(',') > 3:
            score += 1
            
        # Sentences with passive voice (simplified check for Hebrew)
        if any(phrase in sentence for phrase in ['נעשה על ידי', 'בוצע על ידי', 'הוחלט כי']):
            score += 1
            
        # Sentences with redundant words
        redundant_phrases = ['באופן כללי', 'כפי שצוין לעיל', 'יש לציין ש', 'חשוב לציין']
        if any(phrase in sentence for phrase in redundant_phrases):
            score += 1
            
        scored_sentences.append((sentence, score))
    
    # Sort by score (highest first) and take top sentences
    top_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)[:max_sentences]
    
    # Improve each sentence
    improvements = {}
    for sentence, score in top_sentences:
        if score > 0:  # Only improve sentences that need improvement
            improved = improve_sentence(sentence)
            if improved != sentence:  # Only include if actually improved
                improvements[sentence] = improved
    
    return {
        'improvements': improvements,
        'improved_count': len(improvements)
    }

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

def detect_ai_written_text(text: str) -> tuple:
    """
    Detect lines that were likely written by AI.
    Returns a list of AI-written lines and detection metrics.
    """
    print("Running AI text detector...")
    
    # Load whitelist of known human-written lines
    whitelist = set()
    try:
        if os.path.exists('not_ai.txt'):
            with open('not_ai.txt', 'r', encoding='utf-8') as f:
                whitelist = {line.strip() for line in f.readlines() if line.strip()}
            print(f"Loaded {len(whitelist)} whitelisted lines from not_ai.txt")
    except Exception as e:
        print(f"Warning: Could not load not_ai.txt: {e}")
    
    # Split text into lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Features that often indicate AI-generated text
    ai_indicators = []
    ai_written_lines = []
    
    for line in lines:
        if len(line) < 3:  # Skip very short lines
            continue
            
        # Calculate features that may indicate AI-generated text
        score = 0
        reasons = []
        
        # 1. Excessive precision and formality
        words = re.findall(r'[\w\u0590-\u05FF]+', line)  # Include Hebrew Unicode range
        if not words:
            continue
            
        # Check for unnaturally consistent sentence length
        if AI_SENTENCE_LENGTH_MIN <= len(words) <= AI_SENTENCE_LENGTH_MAX:
            score += 1.0
            reasons.append("consistent length")
            
        # 2. Check for repetitive phrases or patterns
        word_set = set(words)
        if len(word_set) / len(words) < AI_LEXICAL_DIVERSITY_THRESHOLD:
            score += 1.0
            reasons.append("low lexical diversity")
            
        # 3. Check for overly balanced punctuation
        punct_count = len(re.findall(r'[.,;:!?،؛]', line))
        if AI_PUNCT_RATIO_MIN <= punct_count / len(line) <= AI_PUNCT_RATIO_MAX:
            score += 1
            reasons.append("balanced punctuation")
            
        # 4. Check for lack of informal elements
        if not re.search(r'(!{2,}|\?{2,}|\.{3,})', line):  # No multiple exclamation/question marks or ellipses
            score += 0.5
            
        # 5. Check for perfect grammatical structure (simplified check)
        if not re.search(r'\b(אה|אממ|כאילו|נו|טוב|יאללה|אז|בקיצור)\b', line.lower()):
            score += 0.5
            
        # 6. Check for generic phrasing
        generic_phrases = [
            'חשוב לציין', 'ניתן לראות', 'לסיכום', 'מצד שני', 
            'לסכם', 'בנוסף', 'יתרה מזאת'
        ]
        
        for phrase in generic_phrases:
            if phrase in line.lower():
                score += 1
                reasons.append(f"generic phrasing: '{phrase}'")
                break
                
        # Store results if score is high enough and line is not in whitelist
        if score >= AI_SCORE_THRESHOLD and line not in whitelist:
            ai_indicators.append({
                'line': line,
                'score': score,
                'reasons': reasons
            })
            ai_written_lines.append(line)
    
    # Save AI-written lines to file
    with open('ai_written_lines.txt', 'w', encoding='utf-8') as f:
        f.write("Lines likely written by AI:\n\n")
        for line in ai_written_lines:
            if not line.strip() or len(line) < 4:
                continue
            f.write(line + '\n\n')
    
    ai_detection_info = {
        'ai_written_lines_count': len(ai_written_lines),
        'total_lines_analyzed': len(lines),
        'percentage_ai': round(len(ai_written_lines) / len(lines) * 100, 2) if lines else 0
    }
    
    return ai_written_lines, ai_detection_info

def analyze_writing_style(text: str) -> dict:
    """
    Analyze writing style metrics for Hebrew text.
    """
    import statistics
    
    print("Analyzing writing style...")
    
    # Split text into lines and sentences
    lines = [line.strip() for line in text.split('\\n') if line.strip()]
    
    # Calculate style metrics for each line
    metrics = []
    for line in lines:
        if len(line) < 3:  # Skip very short lines
            continue
            
        # Calculate basic metrics - use a pattern that works with Hebrew
        words = re.findall(r'[\\w\\u0590-\\u05FF]+', line)  # Include Hebrew Unicode range
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
        return {'error': 'Not enough text to analyze style'}
    
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
        f.write("Lines that don't match the main writing style:\\n\\n")
        for line in outliers:
            f.write(line + '\\n\\n')
    
    return {
        'avg_word_length': mean_word_len,
        'avg_sentence_length': mean_sent_len,
        'avg_punctuation_ratio': mean_punct_ratio,
        'inconsistent_lines_count': len(outliers),
        'total_lines_analyzed': len(metrics)
    }

def main():
    parser = argparse.ArgumentParser(description='Analyze and improve Hebrew text')
    parser.add_argument('file', help='Path to the text file containing the Hebrew story')
    parser.add_argument('--analyze-style', action='store_true',
                        help='Analyze writing style')
    parser.add_argument('--detect-ai', action='store_true',
                        help='Detect lines likely written by AI')
    parser.add_argument('--max-sentences', type=int, default=5,
                        help='Maximum number of sentences to improve')
    parser.add_argument('--min-score', type=int, default=1,
                        help='Minimum improvement score to consider (1-4)')
    parser.add_argument('--model', default='google/mt5-small',
                        help='Model to use for text improvement')
    parser.add_argument('--ai-threshold', type=float, default=AI_SCORE_THRESHOLD,
                        help=f'Threshold score for AI detection (default: {AI_SCORE_THRESHOLD})')
    
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
        style_info = analyze_writing_style(text)
    
    # Detect AI-written text if requested
    if args.detect_ai:
        # Use the threshold from command line arguments
        threshold = args.ai_threshold
        if threshold != AI_SCORE_THRESHOLD:
            print(f"Using custom AI detection threshold: {threshold}")
        
        # Create a modified version of detect_ai_written_text that uses the custom threshold
        def detect_with_custom_threshold(text):
            # Same as detect_ai_written_text but with custom threshold
            print("Running AI text detector...")
            
            # Load whitelist of known human-written lines
            whitelist = set()
            try:
                if os.path.exists('not_ai.txt'):
                    with open('not_ai.txt', 'r', encoding='utf-8') as f:
                        whitelist = {line.strip() for line in f.readlines() if line.strip()}
                    print(f"Loaded {len(whitelist)} whitelisted lines from not_ai.txt")
            except Exception as e:
                print(f"Warning: Could not load not_ai.txt: {e}")
            
            # Split text into lines
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            # Features that often indicate AI-generated text
            ai_indicators = []
            ai_written_lines = []
            
            for line in lines:
                if len(line) < 3:  # Skip very short lines
                    continue
                    
                # Calculate features that may indicate AI-generated text
                score = 0
                reasons = []
                
                # 1. Excessive precision and formality
                words = re.findall(r'[\w\u0590-\u05FF]+', line)  # Include Hebrew Unicode range
                if not words:
                    continue
                    
                # Check for unnaturally consistent sentence length
                if AI_SENTENCE_LENGTH_MIN <= len(words) <= AI_SENTENCE_LENGTH_MAX:
                    score += 1.0
                    reasons.append("consistent length")
                    
                # 2. Check for repetitive phrases or patterns
                word_set = set(words)
                if len(word_set) / len(words) < AI_LEXICAL_DIVERSITY_THRESHOLD:
                    score += 1.0
                    reasons.append("low lexical diversity")
                    
                # 3. Check for overly balanced punctuation
                punct_count = len(re.findall(r'[.,;:!?،؛]', line))
                if AI_PUNCT_RATIO_MIN <= punct_count / len(line) <= AI_PUNCT_RATIO_MAX:
                    score += 1
                    reasons.append("balanced punctuation")
                    
                # 4. Check for lack of informal elements
                if not re.search(r'(!{2,}|\?{2,}|\.{3,})', line):  # No multiple exclamation/question marks or ellipses
                    score += 0.5
                    
                # 5. Check for perfect grammatical structure (simplified check)
                if not re.search(r'\b(אה|אממ|כאילו|נו|טוב|יאללה|אז|בקיצור)\b', line.lower()):
                    score += 0.5
                    
                # 6. Check for generic phrasing
                generic_phrases = [
                    'חשוב לציין', 'ניתן לראות', 'לסיכום', 'מצד שני', 
                    'לסכם', 'בנוסף', 'יתרה מזאת'
                ]
                
                for phrase in generic_phrases:
                    if phrase in line.lower():
                        score += 1
                        reasons.append(f"generic phrasing: '{phrase}'")
                        break
                        
                # Store results if score is high enough and line is not in whitelist
                # Use the custom threshold here
                if score >= threshold and line not in whitelist:
                    ai_indicators.append({
                        'line': line,
                        'score': score,
                        'reasons': reasons
                    })
                    ai_written_lines.append(line)
            
            # Save AI-written lines to file
            with open('ai_written_lines.txt', 'w', encoding='utf-8') as f:
                f.write("Lines likely written by AI:\n\n")
                for line in ai_written_lines:
                    if not line.strip() or len(line) < 4:
                        continue
                    f.write(line + '\n\n')
            
            ai_detection_info = {
                'ai_written_lines_count': len(ai_written_lines),
                'total_lines_analyzed': len(lines),
                'percentage_ai': round(len(ai_written_lines) / len(lines) * 100, 2) if lines else 0
            }
            
            return ai_written_lines, ai_detection_info
        
        # Call the custom function
        ai_lines, ai_info = detect_with_custom_threshold(text)
    
    # Improve text
    print(f"Using model: {args.model}")
    print(f"Looking for sentences with improvement score >= {args.min_score}")
    
    # Override the scoring threshold in improve_text function
    def custom_improve_text(text, max_sentences):
        sentences = re.split(r'([.!?])\s+', text)
        
        # Recombine sentences with their punctuation
        proper_sentences = []
        i = 0
        while i < len(sentences) - 1:
            if i + 1 < len(sentences) and len(sentences[i+1]) == 1 and sentences[i+1] in '.!?':
                proper_sentences.append(sentences[i] + sentences[i+1])
                i += 2
            else:
                proper_sentences.append(sentences[i])
                i += 1
        
        # Filter out very short sentences
        sentences = [s.strip() for s in proper_sentences if len(s.strip()) > 10]
        
        # Score sentences based on complexity and awkwardness
        scored_sentences = []
        for sentence in sentences:
            score = 0
            
            # Long sentences might need improvement
            if len(sentence.split()) > 20:
                score += 1
                
            # Sentences with many commas might be complex
            if sentence.count(',') > 3:
                score += 1
                
            # Sentences with passive voice (simplified check for Hebrew)
            if any(phrase in sentence for phrase in ['נעשה על ידי', 'בוצע על ידי', 'הוחלט כי']):
                score += 1
                
            # Sentences with redundant words
            redundant_phrases = ['באופן כללי', 'כפי שצוין לעיל', 'יש לציין ש', 'חשוב לציין']
            if any(phrase in sentence for phrase in redundant_phrases):
                score += 1
                
            scored_sentences.append((sentence, score))
        
        # Sort by score (highest first) and take top sentences
        top_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)[:max_sentences]
        
        # Improve each sentence
        improvements = {}
        for sentence, score in top_sentences:
            if score >= args.min_score:  # Use the min_score from args
                improved = improve_sentence(sentence, args.model)
                if improved != sentence:  # Only include if actually improved
                    improvements[sentence] = improved
        
        return {
            'improvements': improvements,
            'improved_count': len(improvements)
        }
    
    results = custom_improve_text(text, args.max_sentences)
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*50)
    print(f"Analysis completed in {elapsed_time:.2f} seconds")
    print("="*50)
    
    print(f"\nFound {results['improved_count']} sentences that could be improved:")
    
    for i, (original, improved) in enumerate(results['improvements'].items(), 1):
        print(f"\n{i}. מקורי (Original):")
        print(f"   {original}")
        print(f"\n   משופר (Improved):")
        print(f"   {improved}")
    
    # Save improvements to file
    if results['improved_count'] > 0:
        with open('improved_text.txt', 'w', encoding='utf-8') as f:
            f.write("הצעות לשיפור הטקסט (Text improvement suggestions):\n\n")
            for original, improved in results['improvements'].items():
                f.write(f"מקורי (Original):\n{original}\n\n")
                f.write(f"משופר (Improved):\n{improved}\n\n")
                f.write("-" * 40 + "\n\n")
        print(f"\nהצעות השיפור נשמרו בקובץ (Improvements saved to) improved_text.txt")
    
    # Print style analysis results if available
    if 'style_info' in locals():
        print("\nניתוח סגנון כתיבה (Writing Style Analysis):")
        print(f"- אורך מילה ממוצע (Average word length): {style_info['avg_word_length']:.2f} characters")
        print(f"- אורך משפט ממוצע (Average sentence length): {style_info['avg_sentence_length']:.2f} words")
        print(f"- שורות לא עקביות (Inconsistent lines): {style_info['inconsistent_lines_count']} out of {style_info['total_lines_analyzed']}")
        print(f"- שורות לא עקביות נשמרו לקובץ (Inconsistent lines saved to) bad_lines.txt")
    
    # Print AI detection results if available
    if 'ai_info' in locals():
        print("\nזיהוי טקסט AI (AI Text Detection):")
        print(f"- שורות שנכתבו על ידי AI (AI-written lines): {ai_info['ai_written_lines_count']} out of {ai_info['total_lines_analyzed']} ({ai_info['percentage_ai']}%)")
        print(f"- שורות שנכתבו על ידי AI נשמרו לקובץ (AI-written lines saved to) ai_written_lines.txt")

if __name__ == "__main__":
    main()
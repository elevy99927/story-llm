#!/usr/bin/env python3
"""
Story Analyzer - Analyzes a large text story using local LLM models
Supports both Llama and BERT models for local text analysis
"""

import argparse
import os
import re
import statistics
import random
from typing import Dict, Any, List, Tuple
import time
import pickle
import hashlib

def load_text(file_path: str) -> str:
    """Load text from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        # Try with a different encoding if utf-8 fails
        try:
            with open(file_path, 'r', encoding='cp1255') as file:  # Hebrew Windows encoding
                return file.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='iso-8859-8') as file:  # Another Hebrew encoding
                return file.read()
    except Exception as e:
        print(f"Error loading file: {e}")
        exit(1)

def get_cache_dir():
    """Get or create cache directory for models."""
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.model_cache')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def get_model_hash(model_path):
    """Generate a hash for the model file to use in cache keys."""
    if not model_path or not os.path.exists(model_path):
        return "default"
    
    # Use file modification time and size for quick hash
    stat = os.stat(model_path)
    return f"{os.path.basename(model_path)}_{stat.st_mtime}_{stat.st_size}"

def analyze_with_llama(text: str, model_path: str = None) -> Dict[str, Any]:
    """Analyze text using Llama model."""
    try:
        from llama_cpp import Llama
        
        # Default to a smaller model if none specified
        if not model_path:
            print("No model path specified. Please download a Llama model and provide its path.")
            print("Example: https://huggingface.co/TheBloke/Llama-2-7B-GGUF/tree/main")
            exit(1)
        
        # Create a hash of the text to use as cache key
        text_hash = hashlib.md5(text[:1000].encode()).hexdigest()  # Use first 1000 chars for speed
        model_hash = get_model_hash(model_path)
        cache_key = f"llama_{model_hash}_{text_hash}"
        cache_file = os.path.join(get_cache_dir(), f"{cache_key}.pkl")
        
        # Check if we have cached results
        if os.path.exists(cache_file):
            print(f"Loading cached Llama analysis results...")
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}. Will recompute.")
        
        print(f"Loading Llama model from {model_path}...")
        llm = Llama(model_path=model_path, n_ctx=4096)  # Context size can be adjusted
        
        # Break text into manageable chunks if needed
        chunk_size = 3000  # Adjust based on your model's capabilities
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        results = {
            "summary": "",
            "sentiment": "",
            "themes": [],
            "key_entities": []
        }
        
        # Process first chunk for overall summary
        prompt = f"""
        Analyze the following story excerpt and provide:
        1. A brief summary
        2. Overall sentiment
        3. Main themes
        4. Key characters or entities
        
        Story excerpt:
        {chunks[0]}
        """
        
        print("Analyzing text with Llama...")
        response = llm(prompt, max_tokens=1000)
        results["analysis"] = response["choices"][0]["text"]
        
        # Cache the results
        try:
            cache_file = os.path.join(get_cache_dir(), f"llama_{get_model_hash(model_path)}_{hashlib.md5(text[:1000].encode()).hexdigest()}.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump(results, f)
            print(f"Saved results to cache for faster future analysis")
        except Exception as e:
            print(f"Warning: Could not save to cache: {e}")
        
        return results
        
    except ImportError:
        print("Llama model requires llama-cpp-python package.")
        print("Install with: pip install llama-cpp-python")
        exit(1)
    except Exception as e:
        print(f"Error analyzing with Llama: {e}")
        exit(1)

def analyze_with_bert(text: str) -> Dict[str, Any]:
    """Analyze text using BERT model."""
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        import torch
        
        # Create a hash of the text to use as cache key
        text_hash = hashlib.md5(text[:1000].encode()).hexdigest()  # Use first 1000 chars for speed
        cache_key = f"bert_{text_hash}"
        cache_file = os.path.join(get_cache_dir(), f"{cache_key}.pkl")
        
        # Check if we have cached results
        if os.path.exists(cache_file):
            print(f"Loading cached BERT analysis results...")
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}. Will recompute.")
        
        results = {
            "summary": "",
            "sentiment": "",
            "themes": [],
            "key_entities": []
        }
        
        print("Loading BERT models...")
        
        # Sentiment analysis - use multilingual model with better Hebrew support
        try:
            sentiment_analyzer = pipeline("sentiment-analysis", model="avichr/heBERT_sentiment_analysis")
        except:
            try:
                sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
            except:
                sentiment_analyzer = pipeline("sentiment-analysis")  # Fallback to default
        
        # Named entity recognition - use multilingual model if available
        try:
            ner = pipeline("ner", model="xlm-roberta-large-finetuned-conll03-english")
        except:
            ner = pipeline("ner")  # Fallback to default
        
        # Text summarization
        summarizer = pipeline("summarization")
        
        # Break text into manageable chunks for BERT
        chunk_size = 500  # BERT typically handles 512 tokens max
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        print("Analyzing sentiment...")
        # Analyze sentiment on first few chunks
        sentiments = []
        for i, chunk in enumerate(chunks[:5]):  # Limit to first 5 chunks for speed
            if chunk.strip():
                sentiment = sentiment_analyzer(chunk)[0]
                sentiments.append(sentiment)
                
        # Determine overall sentiment
        pos_count = sum(1 for s in sentiments if s['label'] == 'POSITIVE')
        neg_count = len(sentiments) - pos_count
        results["sentiment"] = "Positive" if pos_count > neg_count else "Negative"
        
        print("Extracting entities...")
        # Extract entities from first few chunks
        entities = {}
        for i, chunk in enumerate(chunks[:3]):  # Limit to first 3 chunks for speed
            if chunk.strip():
                chunk_entities = ner(chunk)
                
                # Group consecutive entity tokens to form complete words
                current_entity = ""
                for entity in chunk_entities:
                    # Remove the underscore prefix that indicates start of word in some tokenizers
                    word = entity['word'].replace('##', '').replace('▁', '')
                    
                    if entity['entity'].startswith('B-'):  # Beginning of entity
                        if current_entity:  # Save previous entity if exists
                            entities[current_entity] = entities.get(current_entity, 0) + 1
                        current_entity = word
                    elif entity['entity'].startswith('I-'):  # Inside/continuation of entity
                        current_entity += word
                    else:  # Not part of a named entity
                        if current_entity:  # Save previous entity if exists
                            entities[current_entity] = entities.get(current_entity, 0) + 1
                            current_entity = ""
                
                # Don't forget the last entity
                if current_entity:
                    entities[current_entity] = entities.get(current_entity, 0) + 1
        
        # Filter out very short entities (likely fragments)
        entities = {k: v for k, v in entities.items() if len(k) >= 2}
        
        # Get top entities
        results["key_entities"] = [k for k, v in sorted(entities.items(), key=lambda x: x[1], reverse=True)[:10]]
        
        print("Generating summary...")
        # Generate summary from first chunk
        if len(chunks[0]) > 100:  # Ensure there's enough text to summarize
            try:
                # Try using mT5 model which handles multiple languages better
                from transformers import MT5ForConditionalGeneration, MT5Tokenizer
                
                print("Using mT5 model for multilingual summarization...")
                tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
                model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
                
                # For Hebrew text, add language hint
                if any('\u0590' <= c <= '\u05FF' for c in chunks[0]):
                    prefix = "summarize Hebrew text: "
                else:
                    prefix = "summarize: "
                
                inputs = tokenizer(prefix + chunks[0], return_tensors="pt", max_length=512, truncation=True)
                summary_ids = model.generate(
                    inputs.input_ids, 
                    max_length=150, 
                    min_length=40, 
                    length_penalty=2.0,
                    num_beams=4  # Add beam search for better results
                )
                results["summary"] = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            except Exception as e:
                print(f"Error with mT5 summarization, falling back to default: {e}")
                # Fallback to original summarizer
                summary = summarizer(chunks[0], max_length=100, min_length=30, do_sample=False)
                results["summary"] = summary[0]['summary_text']
        
        return results
        
    except ImportError:
        print("BERT analysis requires transformers package.")
        print("Install with: pip install transformers torch")
        exit(1)
    except Exception as e:
        print(f"Error analyzing with BERT: {e}")
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
            
        # Calculate basic metrics - use a pattern that works with Hebrew and other languages
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



def detect_ai_written_text(text: str) -> Tuple[List[str], Dict[str, Any]]:
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
        words = re.findall(r'[\w\u0590-\u05FF]+', line)
        if not words:
            continue
            
        # Check for unnaturally consistent sentence length
        if 15 <= len(words) <= 25:
            score += 1
            reasons.append("consistent length")
            
        # 2. Check for repetitive phrases or patterns
        word_set = set(words)
        if len(word_set) / len(words) < 0.7:  # Low lexical diversity
            score += 1
            reasons.append("low lexical diversity")
            
        # 3. Check for overly balanced punctuation
        punct_count = len(re.findall(r'[.,;:!?]', line))
        if 0.08 <= punct_count / len(line) <= 0.12:  # Suspiciously balanced punctuation
            score += 1
            reasons.append("balanced punctuation")
            
        # 4. Check for lack of informal elements
        if not re.search(r'(!{2,}|\?{2,}|\.{3,})', line):  # No multiple exclamation/question marks or ellipses
            score += 0.5
            
        # 5. Check for perfect grammatical structure (simplified check)
        if not re.search(r'\b(um|uh|like|so,|well,|you know)\b', line.lower()):
            score += 0.5
            
        # 6. Check for generic phrasing
        generic_phrases = [
            r'it is important to note', r'as we can see', r'in conclusion', 
            r'on the other hand', r'in summary', r'to summarize'
        ]
        for phrase in generic_phrases:
            if re.search(phrase, line.lower()):
                score += 1
                reasons.append("generic phrasing")
                break
                
        # Store results if score is high enough and line is not in whitelist
        if score >= 2.003 and line not in whitelist:
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
            #skip empty lines   
            if not line.strip():
                continue
            #skip if line have less than 4 chars
            if len(line) < 4:
                continue
            if line.strip():  # Only write non-empty lines
                f.write(line + '\n')
    
    ai_detection_info = {
        'ai_written_lines_count': len(ai_written_lines),
        'total_lines_analyzed': len(lines),
        'percentage_ai': round(len(ai_written_lines) / len(lines) * 100, 2) if lines else 0
    }
    
    return ai_written_lines, ai_detection_info

def main():
    parser = argparse.ArgumentParser(description='Analyze a story text using local LLM models')
    parser.add_argument('file', help='Path to the text file containing the story')
    parser.add_argument('--model', choices=['llama', 'bert'], default='bert',
                        help='Model to use for analysis (default: bert)')
    parser.add_argument('--llama-model', help='Path to Llama model file (required for Llama)')
    parser.add_argument('--analyze-style', action='store_true',
                        help='Analyze writing style and identify inconsistent lines')
    parser.add_argument('--detect-ai', action='store_true',
                        help='Detect lines likely written by AI')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found")
        exit(1)
    
    if args.model == 'llama' and not args.llama_model:
        print("Error: --llama-model path is required when using Llama")
        exit(1)
    
    print(f"Loading text from {args.file}...")
    text = load_text(args.file)
    print(f"Loaded {len(text)} characters ({len(text.split())} words)")
    
    start_time = time.time()
    
    # Analyze writing style if requested
    if args.analyze_style:
        outliers, style_info = analyze_writing_style(text)
    
    # Detect AI-written text if requested
    if args.detect_ai:
        ai_lines, ai_info = detect_ai_written_text(text)
    
    if args.model == 'llama':
        results = analyze_with_llama(text, args.llama_model)
    else:  # bert
        results = analyze_with_bert(text)
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*50)
    print(f"Analysis completed in {elapsed_time:.2f} seconds")
    print("="*50)
    
    if "summary" in results and results["summary"]:
        print(f"\nSummary:\n{results['summary']}")
    
    if "sentiment" in results and results["sentiment"]:
        print(f"\nOverall Sentiment: {results['sentiment']}")
    
    if "key_entities" in results and results["key_entities"]:
        print(f"\nKey Entities: {', '.join(results['key_entities'])}")
    
    if "analysis" in results and results["analysis"]:
        print(f"\nFull Analysis:\n{results['analysis']}")
    
    # Print style analysis results if available
    if 'style_info' in locals():
        print("\nWriting Style Analysis:")
        print(f"- Average word length: {style_info['avg_word_length']:.2f} characters")
        print(f"- Average sentence length: {style_info['avg_sentence_length']:.2f} words")
        print(f"- Inconsistent lines: {style_info['inconsistent_lines_count']} out of {style_info['total_lines_analyzed']}")
        print(f"- Inconsistent lines saved to bad_lines.txt")
    
    # Print AI detection results if available
    if 'ai_info' in locals():
        print("\nAI Text Detection:")
        print(f"- AI-written lines: {ai_info['ai_written_lines_count']} out of {ai_info['total_lines_analyzed']} ({ai_info['percentage_ai']}%)")
        print(f"- AI-written lines saved to ai_written_lines.txt")
    
if __name__ == "__main__":
    main()
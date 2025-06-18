# Hebrew Story Analyzer

כלי לניתוח ושיפור טקסטים בעברית באמצעות מודלים מתקדמים של בינה מלאכותית.

## תכונות

- **ניתוח סגנון כתיבה**: זיהוי שורות שלא תואמות את הסגנון הכללי של הטקסט
- **זיהוי טקסט AI**: זיהוי שורות שנראות כאילו נכתבו על ידי בינה מלאכותית
- **שיפור טקסט**: הצעות לשיפור ניסוח של משפטים מורכבים

## התקנה

1. התקן את סביבת Python (מומלץ Python 3.8 ומעלה)
2. צור סביבת Conda חדשה:
   ```
   conda create -n story_analyzer python=3.9
   conda activate story_analyzer
   ```
3. התקן את הספריות הנדרשות:
   ```
   pip install transformers torch
   ```

## שימוש

### ניתוח סגנון כתיבה

```bash
python hebrew_story_analyzer.py your_story.txt --analyze-style
```

### זיהוי טקסט שנכתב על ידי AI

```bash
python hebrew_story_analyzer.py your_story.txt --detect-ai
```

### שינוי סף הזיהוי של AI

```bash
python hebrew_story_analyzer.py your_story.txt --detect-ai --ai-threshold 1.5
```

### שיפור טקסט

```bash
python hebrew_story_analyzer.py your_story.txt
```

### שילוב כל האפשרויות

```bash
python hebrew_story_analyzer.py your_story.txt --analyze-style --detect-ai
```

## פרמטרים

- `--analyze-style`: ניתוח סגנון כתיבה וזיהוי שורות לא עקביות
- `--detect-ai`: זיהוי שורות שנראות כאילו נכתבו על ידי AI
- `--ai-threshold`: סף הזיהוי של AI (ברירת מחדל: 2.0)
- `--max-sentences`: מספר מקסימלי של משפטים לשיפור (ברירת מחדל: 5)
- `--min-score`: ציון מינימלי לשיפור (ברירת מחדל: 1)
- `--model`: מודל לשימוש לשיפור טקסט (ברירת מחדל: google/mt5-small)

## קבצי פלט

- `bad_lines.txt`: שורות שלא תואמות את הסגנון הכללי של הטקסט
- `ai_written_lines.txt`: שורות שנראות כאילו נכתבו על ידי AI
- `improved_text.txt`: הצעות לשיפור ניסוח של משפטים

## דרישות מערכת

- Python 3.8 ומעלה
- 4GB RAM לפחות
- מעבד מודרני
- חיבור לאינטרנט להורדת מודלים בפעם הראשונה

## רישיון

MIT
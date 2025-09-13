# Análisis simplificado de Programas de Gobierno 2025
import spacy
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

print("🚀 Iniciando análisis de programas de gobierno 2025")

# Cargar modelo de spaCy
nlp = spacy.load('es_core_news_md')
analyzer = SentimentIntensityAnalyzer()

print("✅ Modelos cargados")

# Función para cargar programas
def load_government_programs():
    data_path = Path('data/2025')
    programs = {}
    metadata = {}
    
    for file_path in data_path.glob('*.txt'):
        party_name = file_path.stem
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        party_info = {}
        content_start = 0
        
        for i, line in enumerate(lines):
            if line.startswith('partido:'):
                party_info['partido_completo'] = line.split(':', 1)[1].strip()
            elif line.startswith('candidato_presidente:'):
                party_info['candidato_presidente'] = line.split(':', 1)[1].strip()
            elif line.strip() == '---' and i > 0:
                content_start = i + 1
                break
        
        program_content = '\n'.join(lines[content_start:]).strip()
        programs[party_name] = program_content
        metadata[party_name] = party_info
    
    return programs, metadata

# Cargar datos
programs, metadata = load_government_programs()
print(f"📊 Cargados {len(programs)} programas de gobierno")

# Función de preprocesamiento
def preprocess_text(text, pos_filter=None):
    doc = nlp(text)
    tokens = []
    
    for token in doc:
        if token.is_alpha and len(token.text) > 2 and not token.is_stop:
            if pos_filter and token.pos_ not in pos_filter:
                continue
            tokens.append(token.lemma_.lower())
    
    return tokens

# 1. ANÁLISIS DE FRECUENCIAS
print("\n📈 ANÁLISIS DE FRECUENCIAS")
print("-" * 40)

frequency_analysis = {}
all_tokens = []

for party, program in programs.items():
    tokens = preprocess_text(program, pos_filter=['NOUN', 'ADJ', 'VERB'])
    frequency_analysis[party] = Counter(tokens)
    all_tokens.extend(tokens)
    print(f"{party:8s}: {len(tokens):4d} tokens procesados")

global_frequencies = Counter(all_tokens)
print(f"\n🔝 Top 10 palabras más frecuentes:")
for i, (word, count) in enumerate(global_frequencies.most_common(10), 1):
    print(f"{i:2d}. {word:15s} - {count:3d} veces")

# 2. ANÁLISIS DE SENTIMIENTOS
print("\n😊 ANÁLISIS DE SENTIMIENTOS")
print("-" * 40)

sentiment_results = []

for party, program in programs.items():
    # VADER
    vader_scores = analyzer.polarity_scores(program[:2000])  # Primeros 2000 caracteres
    
    # TextBlob
    blob = TextBlob(program[:2000])
    
    sentiment_results.append({
        'partido': party,
        'vader_compound': vader_scores['compound'],
        'vader_positive': vader_scores['pos'],
        'vader_negative': vader_scores['neg'],
        'textblob_polarity': blob.sentiment.polarity,
        'textblob_subjectivity': blob.sentiment.subjectivity
    })
    
    print(f"{party:8s}: VADER={vader_scores['compound']:+.3f}, TextBlob={blob.sentiment.polarity:+.3f}")

sentiment_df = pd.DataFrame(sentiment_results)

# 3. ESTADÍSTICAS BÁSICAS
print("\n📊 ESTADÍSTICAS BÁSICAS")
print("-" * 40)

stats_results = []
for party, program in programs.items():
    tokens = preprocess_text(program)
    doc = nlp(program)
    entities = [ent.text for ent in doc.ents if len(ent.text) > 2]
    
    stats_results.append({
        'partido': party,
        'total_tokens': len(tokens),
        'unique_tokens': len(set(tokens)),
        'lexical_diversity': len(set(tokens)) / len(tokens) if tokens else 0,
        'total_entities': len(entities),
        'program_length': len(program)
    })
    
    print(f"{party:8s}: {len(tokens):4d} tokens, {len(set(tokens)):3d} únicos, {len(entities):2d} entidades")

stats_df = pd.DataFrame(stats_results)

# COMBINAR RESULTADOS
results_df = sentiment_df.merge(stats_df, on='partido')

print("\n🎯 RESUMEN FINAL")
print("=" * 60)

# Programa más positivo/negativo
most_positive = results_df.loc[results_df['vader_compound'].idxmax()]
most_negative = results_df.loc[results_df['vader_compound'].idxmin()]
print(f"📈 Más positivo: {most_positive['partido']} ({most_positive['vader_compound']:+.3f})")
print(f"📉 Más negativo: {most_negative['partido']} ({most_negative['vader_compound']:+.3f})")

# Programa más extenso/conciso
most_tokens = results_df.loc[results_df['total_tokens'].idxmax()]
least_tokens = results_df.loc[results_df['total_tokens'].idxmin()]
print(f"📚 Más extenso: {most_tokens['partido']} ({most_tokens['total_tokens']} tokens)")
print(f"📄 Más conciso: {least_tokens['partido']} ({least_tokens['total_tokens']} tokens)")

# Mayor diversidad léxica
most_diverse = results_df.loc[results_df['lexical_diversity'].idxmax()]
print(f"🎨 Mayor diversidad léxica: {most_diverse['partido']} ({most_diverse['lexical_diversity']:.3f})")

# Más entidades nombradas
most_entities = results_df.loc[results_df['total_entities'].idxmax()]
print(f"🏷️ Más entidades nombradas: {most_entities['partido']} ({most_entities['total_entities']} entidades)")

# Guardar resultados
results_dir = Path('resultados_analisis')
results_dir.mkdir(exist_ok=True)

results_df.to_csv(results_dir / 'analisis_simple.csv', index=False, encoding='utf-8')

print(f"\n💾 Resultados guardados en: {results_dir / 'analisis_simple.csv'}")
print("✅ Análisis completado exitosamente!")

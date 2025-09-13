# Análisis con visualizaciones de Programas de Gobierno 2025
import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

print("🚀 Iniciando análisis completo con visualizaciones")

# Configuración de visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

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
    # Procesar en chunks para evitar problemas de memoria
    chunks = [text[i:i+50000] for i in range(0, len(text), 50000)]
    tokens = []
    
    for chunk in chunks[:5]:  # Solo los primeros 5 chunks
        try:
            doc = nlp(chunk)
            for token in doc:
                if token.is_alpha and len(token.text) > 2 and not token.is_stop:
                    if pos_filter and token.pos_ not in pos_filter:
                        continue
                    tokens.append(token.lemma_.lower())
        except Exception as e:
            print(f"Error procesando chunk: {e}")
            continue
    
    return tokens

# 1. ANÁLISIS DE FRECUENCIAS
print("\n📈 ANÁLISIS DE FRECUENCIAS")
print("-" * 40)

frequency_analysis = {}
all_tokens = []

for party, program in programs.items():
    print(f"Procesando {party}...")
    tokens = preprocess_text(program, pos_filter=['NOUN', 'ADJ', 'VERB'])
    frequency_analysis[party] = Counter(tokens)
    all_tokens.extend(tokens)
    print(f"{party:8s}: {len(tokens):4d} tokens procesados")

global_frequencies = Counter(all_tokens)
print(f"\n🔝 Top 15 palabras más frecuentes:")
for i, (word, count) in enumerate(global_frequencies.most_common(15), 1):
    print(f"{i:2d}. {word:15s} - {count:3d} veces")

# VISUALIZACIÓN 1: Top palabras más frecuentes
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Top 20 palabras globales
top_words = global_frequencies.most_common(20)
words, counts = zip(*top_words)

axes[0,0].barh(range(len(words)), counts)
axes[0,0].set_yticks(range(len(words)))
axes[0,0].set_yticklabels(words)
axes[0,0].set_title('Top 20 Palabras Más Frecuentes', fontsize=14, fontweight='bold')
axes[0,0].set_xlabel('Frecuencia')
axes[0,0].invert_yaxis()

# WordCloud
try:
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                         max_words=100, colormap='viridis').generate_from_frequencies(global_frequencies)
    axes[0,1].imshow(wordcloud, interpolation='bilinear')
    axes[0,1].axis('off')
    axes[0,1].set_title('Nube de Palabras', fontsize=14, fontweight='bold')
except Exception as e:
    axes[0,1].text(0.5, 0.5, f'Error generando WordCloud:\\n{e}', 
                   transform=axes[0,1].transAxes, ha='center', va='center')
    axes[0,1].set_title('WordCloud (Error)')

# Longitud de programas
program_lengths = [len(preprocess_text(program)) for program in programs.values()]
parties = list(programs.keys())

axes[1,0].bar(parties, program_lengths, color=sns.color_palette("husl", len(parties)))
axes[1,0].set_title('Longitud de Programas por Partido', fontsize=14, fontweight='bold')
axes[1,0].set_ylabel('Número de tokens')
axes[1,0].tick_params(axis='x', rotation=45)

# Distribución de frecuencias
axes[1,1].hist([len(freq.most_common(100)) for freq in frequency_analysis.values()], 
               bins=10, alpha=0.7, color='skyblue')
axes[1,1].set_title('Distribución de Vocabulario', fontsize=14, fontweight='bold')
axes[1,1].set_xlabel('Palabras únicas (top 100)')
axes[1,1].set_ylabel('Número de partidos')

plt.tight_layout()
plt.savefig('resultados_analisis/analisis_frecuencias.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. ANÁLISIS DE SENTIMIENTOS
print("\n😊 ANÁLISIS DE SENTIMIENTOS")
print("-" * 40)

sentiment_results = []

for party, program in programs.items():
    print(f"Analizando sentimientos de {party}...")
    
    # Usar solo los primeros 5000 caracteres para evitar problemas
    sample_text = program[:5000]
    
    # VADER
    vader_scores = analyzer.polarity_scores(sample_text)
    
    # TextBlob
    blob = TextBlob(sample_text)
    
    sentiment_results.append({
        'partido': party,
        'vader_compound': vader_scores['compound'],
        'vader_positive': vader_scores['pos'],
        'vader_negative': vader_scores['neg'],
        'vader_neutral': vader_scores['neu'],
        'textblob_polarity': blob.sentiment.polarity,
        'textblob_subjectivity': blob.sentiment.subjectivity
    })
    
    print(f"{party:8s}: VADER={vader_scores['compound']:+.3f}, TextBlob={blob.sentiment.polarity:+.3f}")

sentiment_df = pd.DataFrame(sentiment_results)

# VISUALIZACIÓN 2: Análisis de sentimientos
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# VADER Compound scores
axes[0,0].bar(sentiment_df['partido'], sentiment_df['vader_compound'], 
              color=['green' if x > 0 else 'red' if x < 0 else 'gray' for x in sentiment_df['vader_compound']])
axes[0,0].set_title('Sentimiento Compuesto (VADER)', fontsize=14, fontweight='bold')
axes[0,0].set_ylabel('Score')
axes[0,0].tick_params(axis='x', rotation=45)
axes[0,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)

# TextBlob scatter
axes[0,1].scatter(sentiment_df['textblob_polarity'], sentiment_df['textblob_subjectivity'], 
                 s=100, alpha=0.7, c=range(len(sentiment_df)), cmap='viridis')
for i, party in enumerate(sentiment_df['partido']):
    axes[0,1].annotate(party, 
                      (sentiment_df['textblob_polarity'].iloc[i], 
                       sentiment_df['textblob_subjectivity'].iloc[i]),
                      xytext=(5, 5), textcoords='offset points', fontsize=8)
axes[0,1].set_xlabel('Polaridad')
axes[0,1].set_ylabel('Subjetividad')
axes[0,1].set_title('Polaridad vs Subjetividad (TextBlob)', fontsize=14, fontweight='bold')
axes[0,1].axhline(y=0.5, color='black', linestyle='--', alpha=0.3)
axes[0,1].axvline(x=0, color='black', linestyle='--', alpha=0.3)

# VADER components (stacked)
bottom_pos = np.zeros(len(sentiment_df))
bottom_neg = sentiment_df['vader_negative'].values

axes[1,0].bar(sentiment_df['partido'], sentiment_df['vader_positive'], 
              label='Positivo', color='green', alpha=0.7)
axes[1,0].bar(sentiment_df['partido'], sentiment_df['vader_neutral'], 
              bottom=sentiment_df['vader_positive'], label='Neutral', color='gray', alpha=0.7)
axes[1,0].bar(sentiment_df['partido'], sentiment_df['vader_negative'], 
              bottom=sentiment_df['vader_positive'] + sentiment_df['vader_neutral'], 
              label='Negativo', color='red', alpha=0.7)
axes[1,0].set_title('Distribución de Sentimientos (VADER)', fontsize=14, fontweight='bold')
axes[1,0].set_ylabel('Proporción')
axes[1,0].tick_params(axis='x', rotation=45)
axes[1,0].legend()

# Comparación métodos
comparison_data = pd.DataFrame({
    'VADER': sentiment_df['vader_compound'],
    'TextBlob': sentiment_df['textblob_polarity']
})
comparison_data.index = sentiment_df['partido']

comparison_data.plot(kind='bar', ax=axes[1,1], alpha=0.8)
axes[1,1].set_title('Comparación de Métodos de Sentimiento', fontsize=14, fontweight='bold')
axes[1,1].set_ylabel('Score')
axes[1,1].tick_params(axis='x', rotation=45)
axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1,1].legend()

plt.tight_layout()
plt.savefig('resultados_analisis/analisis_sentimientos.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. ESTADÍSTICAS Y COMPARACIÓN
print("\n📊 ESTADÍSTICAS COMPARATIVAS")
print("-" * 40)

stats_results = []
for party, program in programs.items():
    print(f"Calculando estadísticas de {party}...")
    
    tokens = preprocess_text(program)
    
    # Procesar entidades en chunks pequeños
    entities = []
    chunks = [program[i:i+10000] for i in range(0, len(program), 10000)]
    for chunk in chunks[:10]:  # Solo primeros 10 chunks
        try:
            doc = nlp(chunk)
            entities.extend([ent.text for ent in doc.ents if len(ent.text) > 2])
        except:
            continue
    
    stats_results.append({
        'partido': party,
        'total_tokens': len(tokens),
        'unique_tokens': len(set(tokens)),
        'lexical_diversity': len(set(tokens)) / len(tokens) if tokens else 0,
        'total_entities': len(entities),
        'program_length': len(program),
        'sentences': program.count('.'),
        'avg_sentence_length': len(tokens) / max(program.count('.'), 1)
    })
    
    print(f"{party:8s}: {len(tokens):4d} tokens, {len(set(tokens)):3d} únicos, {len(entities):2d} entidades")

stats_df = pd.DataFrame(stats_results)

# VISUALIZACIÓN 3: Estadísticas comparativas
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Diversidad léxica vs tokens totales
axes[0,0].scatter(stats_df['total_tokens'], stats_df['lexical_diversity'], 
                 s=100, alpha=0.7, c=range(len(stats_df)), cmap='viridis')
for i, party in enumerate(stats_df['partido']):
    axes[0,0].annotate(party, 
                      (stats_df['total_tokens'].iloc[i], 
                       stats_df['lexical_diversity'].iloc[i]),
                      xytext=(5, 5), textcoords='offset points', fontsize=8)
axes[0,0].set_xlabel('Total de Tokens')
axes[0,0].set_ylabel('Diversidad Léxica')
axes[0,0].set_title('Diversidad Léxica vs Extensión', fontsize=14, fontweight='bold')

# Entidades nombradas
axes[0,1].bar(stats_df['partido'], stats_df['total_entities'], 
              color=sns.color_palette("husl", len(stats_df)))
axes[0,1].set_title('Entidades Nombradas por Partido', fontsize=14, fontweight='bold')
axes[0,1].set_ylabel('Número de Entidades')
axes[0,1].tick_params(axis='x', rotation=45)

# Longitud promedio de oraciones
axes[1,0].bar(stats_df['partido'], stats_df['avg_sentence_length'], 
              color=sns.color_palette("Set2", len(stats_df)))
axes[1,0].set_title('Longitud Promedio de Oraciones', fontsize=14, fontweight='bold')
axes[1,0].set_ylabel('Tokens por Oración')
axes[1,0].tick_params(axis='x', rotation=45)

# Heatmap de correlaciones
combined_df = sentiment_df.merge(stats_df, on='partido')
corr_cols = ['vader_compound', 'textblob_polarity', 'lexical_diversity', 
             'total_tokens', 'total_entities', 'avg_sentence_length']
corr_matrix = combined_df[corr_cols].corr()

im = axes[1,1].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
axes[1,1].set_xticks(range(len(corr_cols)))
axes[1,1].set_yticks(range(len(corr_cols)))
axes[1,1].set_xticklabels([col.replace('_', '\\n') for col in corr_cols], rotation=45, ha='right')
axes[1,1].set_yticklabels([col.replace('_', '\\n') for col in corr_cols])
axes[1,1].set_title('Correlaciones entre Variables', fontsize=14, fontweight='bold')

# Agregar valores de correlación
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        text = axes[1,1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)

plt.colorbar(im, ax=axes[1,1], shrink=0.8)
plt.tight_layout()
plt.savefig('resultados_analisis/estadisticas_comparativas.png', dpi=300, bbox_inches='tight')
plt.show()

# COMBINAR RESULTADOS FINALES
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
least_diverse = results_df.loc[results_df['lexical_diversity'].idxmin()]
print(f"🎨 Mayor diversidad léxica: {most_diverse['partido']} ({most_diverse['lexical_diversity']:.3f})")
print(f"🎯 Menor diversidad léxica: {least_diverse['partido']} ({least_diverse['lexical_diversity']:.3f})")

# Más entidades nombradas
most_entities = results_df.loc[results_df['total_entities'].idxmax()]
print(f"🏷️ Más entidades nombradas: {most_entities['partido']} ({most_entities['total_entities']} entidades)")

# Crear directorio de resultados
results_dir = Path('resultados_analisis')
results_dir.mkdir(exist_ok=True)

# Guardar resultados
results_df.to_csv(results_dir / 'analisis_completo.csv', index=False, encoding='utf-8')

# Generar reporte final
with open(results_dir / 'reporte_final.txt', 'w', encoding='utf-8') as f:
    f.write("ANÁLISIS DE PROGRAMAS DE GOBIERNO 2025\\n")
    f.write("="*60 + "\\n\\n")
    
    f.write("RESUMEN EJECUTIVO:\\n")
    f.write(f"- Total de partidos analizados: {len(programs)}\\n")
    f.write(f"- Total de tokens procesados: {sum(results_df['total_tokens'])}\\n")
    f.write(f"- Palabras más frecuentes: {', '.join([w for w, c in global_frequencies.most_common(5)])}\\n\\n")
    
    f.write("ANÁLISIS DE SENTIMIENTOS:\\n")
    f.write(f"- Programa más positivo: {most_positive['partido']} ({most_positive['vader_compound']:+.3f})\\n")
    f.write(f"- Programa más negativo: {most_negative['partido']} ({most_negative['vader_compound']:+.3f})\\n\\n")
    
    f.write("ESTADÍSTICAS TEXTUALES:\\n")
    f.write(f"- Programa más extenso: {most_tokens['partido']} ({most_tokens['total_tokens']} tokens)\\n")
    f.write(f"- Programa más conciso: {least_tokens['partido']} ({least_tokens['total_tokens']} tokens)\\n")
    f.write(f"- Mayor diversidad léxica: {most_diverse['partido']} ({most_diverse['lexical_diversity']:.3f})\\n")
    f.write(f"- Más entidades nombradas: {most_entities['partido']} ({most_entities['total_entities']} entidades)\\n")

print(f"\n💾 Resultados guardados en: {results_dir}")
print("📁 Archivos generados:")
print("  • analisis_completo.csv (datos completos)")
print("  • reporte_final.txt (resumen ejecutivo)")
print("  • analisis_frecuencias.png (visualización de frecuencias)")
print("  • analisis_sentimientos.png (visualización de sentimientos)")
print("  • estadisticas_comparativas.png (estadísticas comparativas)")
print("✅ Análisis completado exitosamente!")

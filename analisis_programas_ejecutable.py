# Análisis de Programas de Gobierno 2025 con SpaCy
# Script ejecutable independiente

# Importar librerías
import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter, defaultdict
from wordcloud import WordCloud
import os
import re
from pathlib import Path

# Para análisis de sentimientos
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Para topic modeling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from gensim import corpora, models
from gensim.models import LdaModel

# Configuración
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print("✅ Librerías importadas correctamente")

# Cargar modelo de spaCy
nlp = spacy.load('es_core_news_md')

# Configurar analizador de sentimientos VADER
analyzer = SentimentIntensityAnalyzer()

# Cargar modelo transformer para sentimientos en español
sentiment_pipeline = pipeline(
    "sentiment-analysis", 
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    tokenizer="nlptown/bert-base-multilingual-uncased-sentiment"
)

print("✅ Modelos cargados correctamente")
print(f"SpaCy model: {nlp.meta['name']} - {nlp.meta['version']}")

# Función para cargar y procesar los documentos
def load_government_programs():
    """
    Carga todos los programas de gobierno desde la carpeta data/2025
    """
    data_path = Path('/Users/alexojeda/dev/social-data-science/data/2025')
    programs = {}
    metadata = {}
    
    for file_path in data_path.glob('*.txt'):
        party_name = file_path.stem
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extraer metadatos del header
        lines = content.split('\n')
        party_info = {}
        content_start = 0
        
        for i, line in enumerate(lines):
            if line.startswith('partido:'):
                party_info['partido_completo'] = line.split(':', 1)[1].strip()
            elif line.startswith('candidato_presidente:'):
                party_info['candidato_presidente'] = line.split(':', 1)[1].strip()
            elif line.startswith('candidato_vicepresidente:'):
                party_info['candidato_vicepresidente'] = line.split(':', 1)[1].strip()
            elif line.strip() == '---' and i > 0:
                content_start = i + 1
                break
        
        # Extraer el contenido del programa (sin metadatos)
        program_content = '\n'.join(lines[content_start:]).strip()
        
        programs[party_name] = program_content
        metadata[party_name] = party_info
    
    return programs, metadata

# Cargar los datos
programs, metadata = load_government_programs()

print(f"✅ Cargados {len(programs)} programas de gobierno")
print("\nPartidos disponibles:")
for party, info in metadata.items():
    print(f"- {party}: {info.get('partido_completo', 'N/A')}")
    print(f"  Candidato: {info.get('candidato_presidente', 'N/A')}")

# Función de preprocesamiento con spaCy
def preprocess_text(text, remove_stopwords=True, lemmatize=True, pos_filter=None):
    """
    Preprocesa texto usando spaCy
    
    Args:
        text: Texto a procesar
        remove_stopwords: Si remover stopwords
        lemmatize: Si lematizar
        pos_filter: Lista de POS tags a mantener (ej: ['NOUN', 'ADJ', 'VERB'])
    """
    doc = nlp(text)
    
    tokens = []
    for token in doc:
        # Filtrar tokens no deseados
        if token.is_alpha and len(token.text) > 2:
            if remove_stopwords and token.is_stop:
                continue
            if pos_filter and token.pos_ not in pos_filter:
                continue
            
            # Usar lema o texto original
            word = token.lemma_.lower() if lemmatize else token.text.lower()
            tokens.append(word)
    
    return tokens

# Función para extraer entidades nombradas
def extract_entities(text):
    """
    Extrae entidades nombradas del texto
    """
    doc = nlp(text)
    entities = []
    
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'description': spacy.explain(ent.label_)
        })
    
    return entities

print("✅ Funciones de preprocesamiento definidas")

print("\n" + "="*60)
print("🔍 INICIANDO ANÁLISIS DE FRECUENCIAS")
print("="*60)

# Análisis de frecuencias por partido
frequency_analysis = {}
entity_analysis = {}
all_tokens = []
all_entities = []

for party, program in programs.items():
    # Análisis de frecuencias
    tokens = preprocess_text(program, pos_filter=['NOUN', 'ADJ', 'VERB'])
    frequency_analysis[party] = Counter(tokens)
    all_tokens.extend(tokens)
    
    # Análisis de entidades
    entities = extract_entities(program)
    entity_analysis[party] = entities
    all_entities.extend(entities)
    
    print(f"📊 {party}: {len(tokens)} tokens procesados, {len(entities)} entidades encontradas")

# Frecuencias globales
global_frequencies = Counter(all_tokens)
print(f"\n📈 Total: {len(all_tokens)} tokens, {len(set(all_tokens))} únicos")

# Mostrar palabras más frecuentes
print("\n🔝 Top 20 palabras más frecuentes:")
for i, (word, count) in enumerate(global_frequencies.most_common(20), 1):
    print(f"{i:2d}. {word:15s} - {count:4d} veces")

# Función para análisis de sentimientos completo
def analyze_sentiment_comprehensive(text, max_length=512):
    """
    Analiza sentimientos usando múltiples métodos
    """
    # Dividir texto en chunks para modelos con límite de tokens
    chunks = [text[i:i+max_length*4] for i in range(0, len(text), max_length*4)]
    
    results = {
        'vader': {'compound': [], 'pos': [], 'neu': [], 'neg': []},
        'textblob': {'polarity': [], 'subjectivity': []},
        'bert': {'label': [], 'score': []}
    }
    
    for chunk in chunks[:3]:  # Analizar máximo 3 chunks por documento
        if len(chunk.strip()) > 10:
            # VADER
            vader_scores = analyzer.polarity_scores(chunk)
            for key in results['vader']:
                results['vader'][key].append(vader_scores[key])
            
            # TextBlob
            blob = TextBlob(chunk)
            results['textblob']['polarity'].append(blob.sentiment.polarity)
            results['textblob']['subjectivity'].append(blob.sentiment.subjectivity)
            
            # BERT (con manejo de errores)
            try:
                bert_result = sentiment_pipeline(chunk[:512])[0]
                results['bert']['label'].append(bert_result['label'])
                results['bert']['score'].append(bert_result['score'])
            except Exception as e:
                print(f"Error en BERT: {e}")
                results['bert']['label'].append('NEUTRAL')
                results['bert']['score'].append(0.5)
    
    # Promediar resultados
    final_results = {
        'vader_compound': np.mean(results['vader']['compound']) if results['vader']['compound'] else 0,
        'vader_positive': np.mean(results['vader']['pos']) if results['vader']['pos'] else 0,
        'vader_neutral': np.mean(results['vader']['neu']) if results['vader']['neu'] else 0,
        'vader_negative': np.mean(results['vader']['neg']) if results['vader']['neg'] else 0,
        'textblob_polarity': np.mean(results['textblob']['polarity']) if results['textblob']['polarity'] else 0,
        'textblob_subjectivity': np.mean(results['textblob']['subjectivity']) if results['textblob']['subjectivity'] else 0,
        'bert_label': max(set(results['bert']['label']), key=results['bert']['label'].count) if results['bert']['label'] else 'NEUTRAL',
        'bert_score': np.mean(results['bert']['score']) if results['bert']['score'] else 0.5
    }
    
    return final_results

print("\n" + "="*60)
print("😊 INICIANDO ANÁLISIS DE SENTIMIENTOS")
print("="*60)

# Realizar análisis de sentimientos para todos los partidos
sentiment_results = {}

print("🔍 Analizando sentimientos...")
for party, program in programs.items():
    print(f"  Procesando {party}...")
    sentiment_results[party] = analyze_sentiment_comprehensive(program)
    
# Crear DataFrame con resultados
sentiment_df = pd.DataFrame(sentiment_results).T
sentiment_df['partido'] = sentiment_df.index
sentiment_df = sentiment_df.reset_index(drop=True)

print("\n📊 Resultados de análisis de sentimientos:")
print(sentiment_df[['partido', 'vader_compound', 'textblob_polarity', 'textblob_subjectivity']].round(3))

# Preparar datos para topic modeling
def prepare_documents_for_lda(programs):
    """
    Prepara documentos para LDA eliminando palabras muy comunes y muy raras
    """
    documents = []
    doc_names = []
    
    for party, program in programs.items():
        # Procesar texto con filtros más estrictos
        tokens = preprocess_text(program, pos_filter=['NOUN', 'ADJ'])
        
        # Filtrar palabras muy cortas o muy comunes en política
        political_stopwords = {
            'bolivia', 'boliviano', 'boliviana', 'país', 'estado', 'gobierno', 
            'nacional', 'público', 'social', 'económico', 'política', 'político',
            'pueblo', 'ciudadano', 'sociedad', 'desarrollo', 'gestión', 'proceso'
        }
        
        filtered_tokens = [
            token for token in tokens 
            if len(token) > 3 and token not in political_stopwords
        ]
        
        documents.append(filtered_tokens)
        doc_names.append(party)
    
    return documents, doc_names

print("\n" + "="*60)
print("🏷️ INICIANDO TOPIC MODELING")
print("="*60)

# Preparar documentos
documents, doc_names = prepare_documents_for_lda(programs)

# Crear diccionario y corpus para Gensim
dictionary = corpora.Dictionary(documents)

# Filtrar extremos: palabras que aparecen en menos de 2 docs o más del 50% de docs
dictionary.filter_extremes(no_below=2, no_above=0.5)

corpus = [dictionary.doc2bow(doc) for doc in documents]

print(f"📚 Corpus preparado:")
print(f"  - {len(documents)} documentos")
print(f"  - {len(dictionary)} palabras únicas")
print(f"  - {sum(len(doc) for doc in corpus)} tokens totales")

# Entrenar modelo LDA
num_topics = 6  # Número de temas a identificar

print("🤖 Entrenando modelo LDA...")
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=num_topics,
    random_state=42,
    passes=10,
    alpha='auto',
    per_word_topics=True
)

print(f"✅ Modelo LDA entrenado con {num_topics} temas")

# Extraer temas y sus palabras principales
topics = []
for idx in range(num_topics):
    topic_words = lda_model.show_topic(idx, topn=10)
    topics.append({
        'topic_id': idx,
        'words': [word for word, prob in topic_words],
        'probabilities': [prob for word, prob in topic_words],
        'description': ' + '.join([f"{word}({prob:.3f})" for word, prob in topic_words[:5]])
    })

print("\n🏷️ Temas identificados:")
for i, topic in enumerate(topics):
    print(f"\nTema {i}: {', '.join(topic['words'][:5])}")
    print(f"Palabras: {topic['description']}")

# Asignar temas dominantes a cada documento
document_topics = []
for i, doc in enumerate(corpus):
    doc_topics = lda_model.get_document_topics(doc)
    # Obtener tema dominante
    dominant_topic = max(doc_topics, key=lambda x: x[1])
    
    document_topics.append({
        'partido': doc_names[i],
        'dominant_topic_id': dominant_topic[0],
        'dominant_topic_prob': dominant_topic[1],
        'all_topics': doc_topics
    })

# Crear DataFrame de resultados
topic_df = pd.DataFrame(document_topics)
topic_df['dominant_topic_desc'] = topic_df['dominant_topic_id'].apply(
    lambda x: topics[x]['description']
)

print("\n📊 Asignación de temas por partido:")
for _, row in topic_df.iterrows():
    print(f"- {row['partido']:8s} → Tema {row['dominant_topic_id']} ({row['dominant_topic_prob']:.3f})")

# Análisis comparativo integral
comparative_df = sentiment_df.copy()
comparative_df = comparative_df.merge(
    topic_df[['partido', 'dominant_topic_id', 'dominant_topic_prob']], 
    on='partido'
)

# Agregar estadísticas de texto
text_stats = []
for party, program in programs.items():
    tokens = preprocess_text(program)
    entities = extract_entities(program)
    
    text_stats.append({
        'partido': party,
        'total_tokens': len(tokens),
        'unique_tokens': len(set(tokens)),
        'lexical_diversity': len(set(tokens)) / len(tokens) if tokens else 0,
        'total_entities': len(entities),
        'avg_sentence_length': len(tokens) / max(program.count('.'), 1)
    })

text_stats_df = pd.DataFrame(text_stats)
comparative_df = comparative_df.merge(text_stats_df, on='partido')

print("\n" + "="*60)
print("🎯 RESUMEN DEL ANÁLISIS DE PROGRAMAS DE GOBIERNO 2025")
print("="*60)

# 1. Análisis de frecuencias
print("\n📊 ANÁLISIS DE FRECUENCIAS:")
top_5_global = global_frequencies.most_common(5)
print(f"• Palabras más frecuentes globalmente: {', '.join([word for word, _ in top_5_global])}")

longest_program = max([(party, len(preprocess_text(program))) for party, program in programs.items()], key=lambda x: x[1])
shortest_program = min([(party, len(preprocess_text(program))) for party, program in programs.items()], key=lambda x: x[1])
print(f"• Programa más extenso: {longest_program[0]} ({longest_program[1]} tokens)")
print(f"• Programa más conciso: {shortest_program[0]} ({shortest_program[1]} tokens)")

# 2. Análisis de sentimientos
print("\n😊 ANÁLISIS DE SENTIMIENTOS:")
most_positive = comparative_df.loc[comparative_df['vader_compound'].idxmax()]
most_negative = comparative_df.loc[comparative_df['vader_compound'].idxmin()]
print(f"• Programa más positivo (VADER): {most_positive['partido']} ({most_positive['vader_compound']:.3f})")
print(f"• Programa más negativo (VADER): {most_negative['partido']} ({most_negative['vader_compound']:.3f})")

most_subjective = comparative_df.loc[comparative_df['textblob_subjectivity'].idxmax()]
least_subjective = comparative_df.loc[comparative_df['textblob_subjectivity'].idxmin()]
print(f"• Programa más subjetivo: {most_subjective['partido']} ({most_subjective['textblob_subjectivity']:.3f})")
print(f"• Programa más objetivo: {least_subjective['partido']} ({least_subjective['textblob_subjectivity']:.3f})")

# 3. Topic modeling
print("\n🏷️ TOPIC MODELING:")
print(f"• Se identificaron {num_topics} temas principales")
for i, topic in enumerate(topics):
    topic_parties = topic_df[topic_df['dominant_topic_id'] == i]['partido'].tolist()
    print(f"• Tema {i}: {', '.join(topic['words'][:3])} → Partidos: {', '.join(topic_parties)}")

# 4. Características distintivas
print("\n🔍 CARACTERÍSTICAS DISTINTIVAS:")
most_diverse = comparative_df.loc[comparative_df['lexical_diversity'].idxmax()]
least_diverse = comparative_df.loc[comparative_df['lexical_diversity'].idxmin()]
print(f"• Mayor diversidad léxica: {most_diverse['partido']} ({most_diverse['lexical_diversity']:.3f})")
print(f"• Menor diversidad léxica: {least_diverse['partido']} ({least_diverse['lexical_diversity']:.3f})")

most_entities = comparative_df.loc[comparative_df['total_entities'].idxmax()]
print(f"• Más entidades nombradas: {most_entities['partido']} ({most_entities['total_entities']} entidades)")

# 5. Ranking final
ranking_features = ['vader_compound', 'lexical_diversity', 'total_entities', 'dominant_topic_prob']
ranking_df = comparative_df.copy()

# Normalizar features para ranking
for feature in ranking_features:
    ranking_df[f'{feature}_rank'] = ranking_df[feature].rank(ascending=False)

ranking_df['overall_rank'] = ranking_df[[f'{f}_rank' for f in ranking_features]].mean(axis=1)
ranking_df = ranking_df.sort_values('overall_rank')

print("\n🏆 RANKING GENERAL (basado en múltiples métricas):")
top_3_parties = ranking_df.head(3)
for i, (_, party_data) in enumerate(top_3_parties.iterrows(), 1):
    print(f"{i}. {party_data['partido']} (Rank: {party_data['overall_rank']:.2f})")

print("\n" + "="*60)
print("✅ Análisis completado exitosamente")

# Guardar resultados en archivos
import json
from datetime import datetime

# Crear directorio de resultados
results_dir = Path('/Users/alexojeda/dev/social-data-science/resultados_analisis')
results_dir.mkdir(exist_ok=True)

# Guardar resultados del análisis
results = {
    'fecha_analisis': datetime.now().isoformat(),
    'resumen': {
        'total_partidos': len(programs),
        'total_tokens': sum(len(preprocess_text(program)) for program in programs.values()),
        'palabras_mas_frecuentes': dict(global_frequencies.most_common(20)),
        'temas_identificados': {
            f'tema_{i}': {
                'palabras_principales': topic['words'][:10],
                'partidos_asociados': topic_df[topic_df['dominant_topic_id'] == i]['partido'].tolist()
            }
            for i, topic in enumerate(topics)
        }
    },
    'analisis_sentimientos': comparative_df[['partido', 'vader_compound', 'textblob_polarity', 'textblob_subjectivity']].to_dict('records'),
    'topic_modeling': topic_df[['partido', 'dominant_topic_id', 'dominant_topic_prob']].to_dict('records'),
    'estadisticas_texto': text_stats_df.to_dict('records')
}

# Guardar en JSON
with open(results_dir / 'analisis_programas_gobierno_2025.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# Guardar DataFrames en CSV
comparative_df.to_csv(results_dir / 'analisis_comparativo.csv', index=False, encoding='utf-8')
sentiment_df.to_csv(results_dir / 'analisis_sentimientos.csv', index=False, encoding='utf-8')
topic_df.to_csv(results_dir / 'topic_modeling.csv', index=False, encoding='utf-8')

print(f"\n💾 Resultados guardados en: {results_dir}")
print("📁 Archivos generados:")
print("  • analisis_programas_gobierno_2025.json (resumen completo)")
print("  • analisis_comparativo.csv (datos comparativos)")
print("  • analisis_sentimientos.csv (resultados de sentimientos)")
print("  • topic_modeling.csv (resultados de temas)")

print("\n🎉 ¡ANÁLISIS COMPLETO!")

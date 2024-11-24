from flask import Flask, request, jsonify
import pandas as pd
import os
import re
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import hnswlib
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

results = None
grant_embeddings = None
cluster_centers_df = None
qa_pipeline = None
embedding_model = None
num_clusters = 20

def preprocess_content(content):
    if not isinstance(content, str):
        return content  
    content_cleaned = re.sub(r"[^\w\s]", " ", content)  
    content_cleaned = re.sub(r"\s+", " ", content_cleaned) 
    return content_cleaned.strip()

def process_csv_and_fetch_content(csv_file, content_dir):
    df = pd.read_csv(csv_file)
    content_list = []
    for idx in df.index:
        file_path = os.path.join(content_dir, f"{idx + 1}.txt")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = None  
        processed_content = preprocess_content(content)
        content_list.append(processed_content)
    df["Content"] = content_list
    return df

def extract_features_from_content(content, qa_pipeline):
    if not isinstance(content, str) or not content.strip():
        return {
            'Сумма гранта': None,
            'Сфера интересов': None,
            'Критерии': None,
            'Тип': None,
            'Регион': None,
            'Документы': None,
            'Форма подачи': None,
            'Тип заявителя': None,
        }
    questions = {
        'Сумма гранта': "Какова сумма гранта?",
        'Сфера интересов': "Какая сфера интересов указана?",
        'Критерии': "Какие критерии для участия?",
        'Тип': "Какой тип гранта указан?",
        'Регион': "На какой регион распространяется грант?",
        'Документы': "Какие документы требуются для подачи заявки?",
        'Форма подачи': "Какая форма подачи заявки указана?",
        'Тип заявителя': "Кто может подавать заявку на грант?",
    }
    features = {}
    for feature, question in questions.items():
        try:
            answer = qa_pipeline(question=question, context=content)
            features[feature] = answer['answer']
        except:
            features[feature] = None
    return features

def process_and_extract_features(results, qa_pipeline):
    extracted_features_list = []
    for idx, row in results.iterrows():
        content = row['Content']
        features = extract_features_from_content(content, qa_pipeline)
        extracted_features_list.append(features)
    features_df = pd.DataFrame(extracted_features_list)
    return pd.concat([results.reset_index(drop=True), features_df], axis=1)

def generate_embeddings(df, embedding_model):
    text_features = ['title', 'Content']
    df['CombinedText'] = df[text_features].fillna("").agg(" ".join, axis=1)
    embeddings = embedding_model.encode(df['CombinedText'].tolist(), show_progress_bar=True)
    return embeddings

def cluster_embeddings(embeddings, num_clusters=20):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    return cluster_labels, cluster_centers

def generate_startup_embeddings(startups_df, embedding_model):
    startups_df['CombinedText'] = startups_df.apply(
        lambda row: f"Стадия: {row['Стадия стартапа']}. Отрасль: {row['Отрасль']}. "
                    f"Инновационный фокус: {row['Инновационный фокус']}. "
                    f"Описание: {row['Описание']}.",
        axis=1
    )
    embeddings = embedding_model.encode(startups_df['CombinedText'].tolist(), show_progress_bar=True)
    return embeddings

def map_startups_to_clusters(startups_df_with_clusters, cluster_centers_df):
    cluster_centers = cluster_centers_df.values  
    cluster_names = cluster_centers_df.index.tolist()  
    closest_clusters = []
    for idx, row in startups_df_with_clusters.iterrows():
        startup_embedding = np.array(row['Embeddings'])
        similarities = cosine_similarity(
            startup_embedding.reshape(1, -1),
            cluster_centers
        )[0]
        closest_cluster_idx = np.argmax(similarities)
        closest_cluster_name = cluster_names[closest_cluster_idx]
        closest_clusters.append(closest_cluster_name)
    startups_df_with_clusters['Closest Cluster'] = closest_clusters
    return startups_df_with_clusters

def find_top_clusters_and_grants(startups_df_with_clusters, cluster_centers_df, grant_embeddings, grants_df, top_n_clusters=5, top_n_grants=5):
    cluster_centers = cluster_centers_df.values
    cluster_names = cluster_centers_df.index.tolist()
    results = {}
    for _, startup in startups_df_with_clusters.iterrows():
        startup_id = startup['ID стартапа']
        startup_embedding = np.array(startup['Embeddings'])
        cluster_similarities = cosine_similarity(
            startup_embedding.reshape(1, -1),
            cluster_centers
        )[0]
        top_clusters_indices = np.argsort(cluster_similarities)[-top_n_clusters:][::-1]
        selected_grants = []
        for cluster_idx in top_clusters_indices:
            cluster_grant_indices = grants_df[grants_df['Cluster'] == cluster_idx].index
            if len(cluster_grant_indices) == 0:
                continue
            cluster_grant_embeddings = grant_embeddings[cluster_grant_indices]
            grant_similarities = cosine_similarity(
                startup_embedding.reshape(1, -1),
                cluster_grant_embeddings
            )[0]
            top_grant_indices = np.argsort(grant_similarities)[-top_n_grants:][::-1]
            selected_grants.extend([grants_df.iloc[idx] for idx in cluster_grant_indices[top_grant_indices]])
        results[startup_id] = selected_grants
    return results

def initialize():
    global results, grant_embeddings, cluster_centers_df, qa_pipeline, embedding_model, num_clusters
    results = process_csv_and_fetch_content("parsed_data.csv", "products_content")
    qa_pipeline = pipeline("question-answering", model="Den4ikAI/rubert_large_squad_2")
    results = process_and_extract_features(results, qa_pipeline)
    embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    grant_embeddings = generate_embeddings(results, embedding_model)
    results['Embeddings'] = grant_embeddings.tolist()
    cluster_labels, cluster_centers = cluster_embeddings(grant_embeddings, num_clusters=num_clusters)
    results['Cluster'] = cluster_labels
    cluster_centers_df = pd.DataFrame(
        cluster_centers,
        columns=[f"dim_{i}" for i in range(cluster_centers.shape[1])],
        index=[f"Cluster_{i}" for i in range(num_clusters)]
    )

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    global results, grant_embeddings, cluster_centers_df, embedding_model
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Нет данных'}), 400
    expected_keys = ['ID стартапа', 'Стадия стартапа', 'Отрасль', 'Выручка', 'Необходимое финансирование', 'Локация',
                     'Лет на рынке', 'Размер команды', 'Инновационный фокус', 'Описание', 'История участия']
    if not all(key in data for key in expected_keys):
        return jsonify({'error': 'Отсутствуют необходимые поля в данных'}), 400
    startup_df = pd.DataFrame([data])
    startup_embeddings = generate_startup_embeddings(startup_df, embedding_model)
    startup_df['Embeddings'] = startup_embeddings.tolist()
    startup_df = map_startups_to_clusters(startup_df, cluster_centers_df)
    grant_recommendations = find_top_clusters_and_grants(
        startups_df_with_clusters=startup_df,
        cluster_centers_df=cluster_centers_df,
        grant_embeddings=np.array(results['Embeddings'].tolist()),
        grants_df=results,
        top_n_clusters=5,
        top_n_grants=5
    )
    startup_id = data['ID стартапа']
    grants = grant_recommendations.get(startup_id, [])
    grant_list = []
    for grant in grants:
        grant_info = {
            'title': grant['title'],
            'url': grant['url'],
            'status': grant['status'],
            'timing': grant['timing'],
            'Content': grant['Content'],
        }
        grant_list.append(grant_info)
    return jsonify({'recommendations': grant_list})

if __name__ == '__main__':
    initialize()
    app.run(host='0.0.0.0', port=5000)
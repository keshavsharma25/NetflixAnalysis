import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def create_cosine_sim(df):

    count = CountVectorizer(stop_words="english", ngram_range=(1, 5))
    count_matrix = count.fit_transform(df["overview"])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    indices = pd.Series(df.index, index=df["title"])

    return cosine_sim, indices


def get_recommendations(df, title, indices, cosine_sim, top_k=5):

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[0:20]
    movie_indices = [i[0] for i in sim_scores if i[0] != idx]

    return (
        df.iloc[movie_indices]
        .sort_values(["imdb_votes", "imdb_score"], ascending=False)[
            ["title", "description", "genres", "imdb_score"]
        ]
        .reset_index(drop=True)
        .head(top_k)
    )

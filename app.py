import math
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache

import gradio as gr
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


NUM_DOCS = int(os.environ.get("NUM_DOCS", "5000"))
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

STOPWORDS = set("""
a an the and or but if while is are was were be been being to of in on for from with as by at this that these those
it its into about not no can could should would will just do does did have has had i you he she they we them his her their our
""".split())


def preprocess(text: str):
    text = text.lower()
    tokens = re.findall(r"[a-z]+", text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


@lru_cache(maxsize=1)
def load_corpus_and_indexes():
    dataset = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
    )

    documents = list(dataset.data[:NUM_DOCS])
    labels = np.array(dataset.target[:NUM_DOCS])
    target_names = list(dataset.target_names)

    inverted_index = defaultdict(dict)
    doc_term_freqs = {}

    for doc_id, text in enumerate(documents):
        tokens = preprocess(text)
        term_freq = Counter(tokens)
        doc_term_freqs[doc_id] = term_freq

        for term, freq in term_freq.items():
            inverted_index[term][doc_id] = freq

    total_docs = len(doc_term_freqs)
    idf = {}
    for term, postings in inverted_index.items():
        idf[term] = math.log10(total_docs / len(postings))

    doc_vectors = {}
    for doc_id, term_freq in doc_term_freqs.items():
        vector = {}
        for term, tf in term_freq.items():
            vector[term] = tf * idf[term]

        norm = math.sqrt(sum(weight ** 2 for weight in vector.values()))
        if norm != 0:
            for term in vector:
                vector[term] /= norm

        doc_vectors[doc_id] = vector

    clean_documents = []
    for doc in documents:
        cleaned = doc.replace("\n", " ").strip()
        clean_documents.append(cleaned if cleaned else "empty document")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    doc_embeddings = embedding_model.encode(
        clean_documents,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    return {
        "documents": documents,
        "labels": labels,
        "target_names": target_names,
        "inverted_index": inverted_index,
        "idf": idf,
        "doc_vectors": doc_vectors,
        "embedding_model": embedding_model,
        "doc_embeddings": doc_embeddings,
    }


@lru_cache(maxsize=1)
def load_llm():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch

    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    llm_model = llm_model.to(device)
    return tokenizer, llm_model, device


def get_category_id(category_name: str) -> int:
    data = load_corpus_and_indexes()
    return data["target_names"].index(category_name)


def tfidf_search(query: str, top_k: int = 10):
    data = load_corpus_and_indexes()
    idf = data["idf"]
    inverted_index = data["inverted_index"]
    doc_vectors = data["doc_vectors"]

    query_tokens = preprocess(query)
    query_tf = Counter(query_tokens)

    query_vector = {}
    for term, tf in query_tf.items():
        if term in idf:
            query_vector[term] = tf * idf[term]

    query_norm = math.sqrt(sum(weight ** 2 for weight in query_vector.values()))
    if query_norm != 0:
        for term in query_vector:
            query_vector[term] /= query_norm

    candidate_docs = set()
    for term in query_vector:
        if term in inverted_index:
            candidate_docs.update(inverted_index[term].keys())

    scores = {}
    for doc_id in candidate_docs:
        score = 0
        for term, q_weight in query_vector.items():
            score += q_weight * doc_vectors[doc_id].get(term, 0)
        scores[doc_id] = score

    ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_results[:top_k]


def semantic_search(query: str, top_k: int = 10):
    data = load_corpus_and_indexes()
    embedding_model = data["embedding_model"]
    doc_embeddings = data["doc_embeddings"]

    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [(int(doc_id), float(similarities[doc_id])) for doc_id in top_indices]


def hybrid_search(query: str, top_k: int = 10, alpha: float = 0.5):
    tfidf_results = tfidf_search(query, top_k=100)
    semantic_results = semantic_search(query, top_k=100)

    tfidf_scores = {doc_id: score for doc_id, score in tfidf_results}
    semantic_scores = {doc_id: score for doc_id, score in semantic_results}

    all_doc_ids = set(tfidf_scores.keys()).union(set(semantic_scores.keys()))

    max_tfidf = max(tfidf_scores.values()) if tfidf_scores else 1
    max_semantic = max(semantic_scores.values()) if semantic_scores else 1

    hybrid_scores = {}
    for doc_id in all_doc_ids:
        normalized_tfidf = tfidf_scores.get(doc_id, 0) / max_tfidf if max_tfidf != 0 else 0
        normalized_semantic = semantic_scores.get(doc_id, 0) / max_semantic if max_semantic != 0 else 0
        hybrid_scores[doc_id] = alpha * normalized_tfidf + (1 - alpha) * normalized_semantic

    ranked_results = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_results[:top_k]


def generate_query_expansion(query: str):
    tokenizer, llm_model, device = load_llm()

    prompt = (
        "Generate 8 useful search keywords related to this query. "
        "Return only keywords, no full sentences. Query: " + query
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(device)

    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def llm_expanded_search(query: str, top_k: int = 10):
    try:
        generated_text = generate_query_expansion(query)
    except Exception as exc:
        generated_text = f"LLM expansion unavailable on this runtime: {exc}"
        return tfidf_search(query, top_k=top_k), generated_text, query

    expanded_query = (query + " ") * 7 + generated_text
    results = tfidf_search(expanded_query, top_k=top_k)
    return results, generated_text, expanded_query


def run_method(query: str, method: str, top_k: int, alpha: float):
    generated_expansion = ""

    if method == "TF-IDF":
        results = tfidf_search(query, top_k=top_k)
    elif method == "Semantic":
        results = semantic_search(query, top_k=top_k)
    elif method == "Hybrid":
        results = hybrid_search(query, top_k=top_k, alpha=alpha)
    elif method == "LLM Query Expansion":
        results, generated_expansion, _ = llm_expanded_search(query, top_k=top_k)
    else:
        results = []

    return results, generated_expansion




def format_results(query, method, top_k, alpha):
    if not query or query.strip() == "":
        return "Please enter a search query."

    data = load_corpus_and_indexes()
    documents = data["documents"]
    labels = data["labels"]
    target_names = data["target_names"]

    top_k = int(top_k)
    alpha = float(alpha)

    results, generated_expansion = run_method(query, method, top_k, alpha)

    output = "## Ranked Search Results\n\n"

    if generated_expansion:
        output += f"**Generated Expansion:** {generated_expansion}\n\n---\n\n"

    for rank, (doc_id, score) in enumerate(results, start=1):
        snippet = documents[doc_id][:700].replace("\n", " ")

        output += f"""
### {rank}. Document ID: {doc_id}

**Score:** {round(float(score), 4)}  
**Category:** {target_names[int(labels[doc_id])]}  
**Method:** {method}

{snippet}...

---
"""

    return output

def evaluate_method_query(query, relevant_category, method, top_k, alpha):
    if not query or query.strip() == "":
        return "Please enter a query.", "No metrics available."

    data = load_corpus_and_indexes()
    documents = data["documents"]
    labels = data["labels"]
    target_names = data["target_names"]

    top_k = int(top_k)
    alpha = float(alpha)
    results, generated_expansion = run_method(query, method, top_k, alpha)
    relevant_cat_id = get_category_id(relevant_category)

    relevant_found = 0
    precision_sum = 0

    results_md = "## Search Results with Relevance Labels\n\n"

    if generated_expansion:
        results_md += f"**Generated Expansion:** {generated_expansion}\n\n---\n\n"

    for rank, (doc_id, score) in enumerate(results, start=1):
        is_relevant = int(labels[doc_id]) == relevant_cat_id

        if is_relevant:
            relevant_found += 1
            precision_sum += relevant_found / rank

        snippet = documents[doc_id][:700].replace("\n", " ")
        relevant_text = "Yes" if is_relevant else "No"

        results_md += f"""
### {rank}. Document ID: {doc_id}

**Score:** {round(float(score), 4)}  
**Category:** {target_names[int(labels[doc_id])]}  
**Relevant?:** {relevant_text}  
**Method:** {method}

{snippet}...

---
"""

    total_relevant_docs = int(sum(1 for label in labels if int(label) == relevant_cat_id))

    precision = relevant_found / top_k if top_k > 0 else 0
    recall = relevant_found / total_relevant_docs if total_relevant_docs > 0 else 0
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    ap = 0 if relevant_found == 0 else precision_sum / relevant_found

    metrics_md = f"""
## Evaluation Metrics

| Metric | Score |
|---|---:|
| Precision@{top_k} | {round(precision, 4)} |
| Recall@{top_k} | {round(recall, 4)} |
| F1@{top_k} | {round(f1, 4)} |
| Average Precision@{top_k} | {round(ap, 4)} |
| Relevant Found | {relevant_found} |
| Total Relevant Docs | {total_relevant_docs} |
"""

    return results_md, metrics_md

    

def compare_all_methods(query, relevant_category, top_k, alpha):
    if not query or query.strip() == "":
        return pd.DataFrame(columns=["Method", "Precision", "Recall", "F1", "Average Precision", "Relevant Found"])

    methods = ["TF-IDF", "Semantic", "Hybrid", "LLM Query Expansion"]
    rows = []

    for method in methods:
        _, metrics_df = evaluate_method_query(query, relevant_category, method, top_k, alpha)
        metric_dict = dict(zip(metrics_df["Metric"], metrics_df["Score"]))

        rows.append({
            "Method": method,
            "Precision": metric_dict.get(f"Precision@{int(top_k)}", 0),
            "Recall": metric_dict.get(f"Recall@{int(top_k)}", 0),
            "F1": metric_dict.get(f"F1@{int(top_k)}", 0),
            "Average Precision": metric_dict.get(f"Average Precision@{int(top_k)}", 0),
            "Relevant Found": metric_dict.get("Relevant Found", 0),
        })

    return pd.DataFrame(rows)

    

def get_status():
    data = load_corpus_and_indexes()
    return (
        f"Loaded {len(data['documents'])} documents from 20 Newsgroups. "
        f"Categories: {len(data['target_names'])}. "
        f"Embedding model: {EMBEDDING_MODEL_NAME}."
    )


def build_demo():
    # Load once at startup so the interface is ready after Space build/start.
    data = load_corpus_and_indexes()
    target_names = data["target_names"]

    with gr.Blocks(title="Interactive Local IR Search Engine") as demo:
        gr.Markdown("# Interactive Local Text Document Search Engine")
        gr.Markdown(
            "Search a local text corpus using TF-IDF, semantic embeddings, hybrid retrieval, "
            "and LLM-based query expansion. You can also evaluate retrieval quality using category-based relevance judgments."
        )

        with gr.Accordion("System Status", open=False):
            status_box = gr.Textbox(value=get_status(), label="Loaded Corpus", interactive=False)

        with gr.Tab("Search"):
            query_input = gr.Textbox(label="Enter Search Query", placeholder="Example: space nasa moon")
            method_input = gr.Radio(
                choices=["TF-IDF", "Semantic", "Hybrid", "LLM Query Expansion"],
                value="TF-IDF",
                label="Search Method",
            )
            top_k_input = gr.Slider(minimum=5, maximum=20, value=10, step=1, label="Number of Results")
            alpha_input = gr.Slider(
                minimum=0,
                maximum=1,
                value=0.5,
                step=0.1,
                label="Hybrid Weight Alpha: 1 = TF-IDF, 0 = Semantic",
            )
            search_button = gr.Button("Search")
            search_output = gr.Markdown(label="Ranked Search Results")

            search_button.click(
                fn=format_results,
                inputs=[query_input, method_input, top_k_input, alpha_input],
                outputs=search_output,
            )

        with gr.Tab("Evaluate Your Query"):
            eval_query_input = gr.Textbox(label="Enter Query to Evaluate", placeholder="Example: space nasa moon")
            category_dropdown = gr.Dropdown(choices=target_names, value="sci.space", label="Select Relevant Category")
            eval_method_input = gr.Radio(
                choices=["TF-IDF", "Semantic", "Hybrid", "LLM Query Expansion"],
                value="TF-IDF",
                label="Evaluation Method",
            )
            eval_top_k_input = gr.Slider(minimum=5, maximum=20, value=10, step=1, label="Evaluate Top K Results")
            eval_alpha_input = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.1, label="Hybrid Weight Alpha")
            eval_button = gr.Button("Evaluate Query")
            eval_results_output = gr.Markdown(label="Search Results with Relevance Labels")
            eval_metrics_output = gr.Markdown(label="Evaluation Metrics")
            gr.Markdown(
                "Note: For evaluation, the selected category is treated as the ground-truth relevant class. "
                "Results from that category are marked as relevant."
            )

            eval_button.click(
                fn=evaluate_method_query,
                inputs=[eval_query_input, category_dropdown, eval_method_input, eval_top_k_input, eval_alpha_input],
                outputs=[eval_results_output, eval_metrics_output],
            )

        with gr.Tab("Compare Methods"):
            compare_query_input = gr.Textbox(label="Enter Query", placeholder="Example: space nasa moon")
            compare_category_dropdown = gr.Dropdown(choices=target_names, value="sci.space", label="Select Relevant Category")
            compare_top_k_input = gr.Slider(minimum=5, maximum=20, value=10, step=1, label="Top K")
            compare_alpha_input = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.1, label="Hybrid Weight Alpha")
            compare_button = gr.Button("Compare Methods")
            compare_output = gr.Dataframe(label="Method Comparison", wrap=True)

            compare_button.click(
                fn=compare_all_methods,
                inputs=[compare_query_input, compare_category_dropdown, compare_top_k_input, compare_alpha_input],
                outputs=compare_output,
            )

    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

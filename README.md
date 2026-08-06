# Airbnb-Review-Analytics-Thematic-Recommendation-System
https://airbnb-review-analytics-thematic-recommendation-system-rw3jxx7.streamlit.app/
## Project Overview

This project develops a theme-based recommendation system that analyzes large-scale customer reviews using Natural Language Processing (NLP) and topic modeling. The system leverages BERTopic, BERT embeddings, and clustering techniques to identify key experience themes from unstructured text and recommend relevant listings aligned with user preferences.

The solution transforms unstructured review data into actionable insights that support decision-making for both users and business stakeholders.

In this project, over 500,000+ customer reviews across 6,000+ listings were analyzed to uncover meaningful patterns in customer sentiment and preferences.

## Business Problem

Online platforms such as Airbnb contain massive volumes of user-generated reviews. While these reviews provide valuable feedback, extracting meaningful insights manually is impractical at scale.

Traditional keyword-based search systems often fail to capture semantic meaning, resulting in:

Poor listing discovery

Limited personalization

Missed business insights

Reduced customer satisfaction

This project addresses these challenges by building a semantic recommendation engine that identifies themes from customer reviews and recommends listings based on user preferences.

## Objectives

Identify hidden themes from large-scale review data

Build a recommendation system using semantic similarity

Improve listing discovery using NLP techniques

Generate actionable insights for hosts and stakeholders

Evaluate topic quality using quantitative metrics

## Dataset

Source: Airbnb Reviews Dataset
Volume:

541,000+ reviews

6,000+ listings

Text-based customer feedback

Data Fields Used:

Review text

Listing ID

Rating

Review date

Location

## System Architecture

The recommendation system follows a structured NLP pipeline:

Data Collection
→ Text Preprocessing
→ BERT Embeddings
→ BERTopic Modeling
→ Topic Evaluation
→ Similarity Matching
→ Recommendation Engine

## Methodology
### 1. Data Preprocessing

The raw review text was cleaned and standardized before modeling.

Steps included:

Lowercasing text

Removing punctuation and stopwords

Tokenization

Lemmatization

Removing noise and duplicate reviews

Libraries used:

Pandas

NLTK

SpaCy

### 2. Feature Engineering

Text embeddings were generated using transformer-based models.

Techniques used:

BERT embeddings

Sentence Transformers

Dimensionality reduction using UMAP

Density-based clustering using HDBSCAN

These steps enabled semantic grouping of similar reviews.

### 3. Topic Modeling using BERTopic

BERTopic was used to identify meaningful themes from customer reviews.

The model combines:

BERT embeddings

UMAP dimensionality reduction

HDBSCAN clustering

Class-based TF-IDF

This approach produces interpretable topics representing customer experience themes.

Example themes discovered:

Cleanliness

Location convenience

Parking availability

Safety

Host responsiveness

Noise levels

### 4. Recommendation Engine

The recommendation system identifies listings that best match user preferences.

Process:

Convert reviews into embeddings

Identify dominant themes

Compute similarity scores

Recommend relevant listings

Similarity technique:

Cosine similarity

### 5. Model Evaluation

The performance of the topic model and recommendation system was evaluated using quantitative metrics.

Topic Modeling Metrics

Topic Coherence Score

Topic Diversity Score

These metrics validate the quality and interpretability of discovered themes.

## Key Results

Identified 12 meaningful customer experience themes from large-scale review data using BERTopic, enabling structured analysis of unstructured text feedback.

Improved recommendation relevance using semantic similarity, delivering listings similar to the current listing and aligned with user search preferences.

Generated top themes for each listing based on aggregated user reviews, providing interpretable insights into customer experience patterns.

Enabled scalable analysis of unstructured text data across 500K+ reviews, supporting efficient large-scale text analytics.

Produced actionable insights for hosts and stakeholders to identify recurring issues and improve service quality.

Enhanced personalized listing discovery by recommending contextually relevant properties based on user preferences and review themes.

## Business Impact

This system enables organizations to:

Improve customer experience through personalized recommendations

Identify recurring customer issues from reviews

Support data-driven product and service improvements

Enhance platform engagement and retention

Scale insight generation from unstructured feedback

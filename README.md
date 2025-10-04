# Twitter Data Analysis & Natural Language Processing Project

This repository contains three complementary Jupyter notebooks designed to explore, analyze, and model Twitter data.  
The main goal of this project is to **understand user behavior** and **analyze textual content** through statistical methods, machine learning, and large language models (LLMs).

---

## Project Overview

The project is divided into three main analytical components:

1. **User-level statistical analysis** — exploring quantitative patterns of user behavior (followers, activity, engagement).  
2. **Natural Language Processing (NLP)** — performing text cleaning, tokenization, and sentiment classification of tweets.  
3. **Large Language Models (LLMs)** — leveraging pre-trained language models to capture contextual understanding of tweets.

Each notebook builds upon the previous one, moving from structured data analysis to text analytics and finally to advanced generative AI techniques.

---

## 📂 Repository Structure

| Notebook | Description | Main Methods |
|-----------|--------------|--------------|
| **`main code of analysis.ipynb`** | Statistical analysis and segmentation of Twitter users. | Data cleaning, PCA, K-Means clustering, correlation analysis, visualization. |
| **`NLP_code.ipynb`** | Text classification and sentiment analysis of tweets. | Text preprocessing, tokenization, padding, LSTM model, sentiment prediction. |
| **`LLM_code.ipynb`** | Experiments with large language models for contextual tweet understanding. | Prompt engineering, zero/few-shot learning, qualitative evaluation. |

---

## 1️⃣ `main code of analysis.ipynb` — User Behavior Segmentation

This notebook performs a **quantitative analysis of Twitter users** based on their engagement and profile statistics.

### Main Steps
1. **Data Preprocessing**
   - Aggregation of user-level statistics: followers, following, listed count, tweet count.
   - Computation of engagement statistics (mean, std, median, quartiles) for likes, replies, retweets, quotes, mentions, and hashtags.
   - Addition of temporal metrics: average time between tweets.

2. **Exploratory Data Analysis**
   - Correlation matrix analysis to identify relationships between user attributes.
   - Visualization of distributions and dependencies between variables.

3. **Principal Component Analysis (PCA)**
   - Dimensionality reduction on both user and engagement metrics.
   - Axis interpretation:
     - **PC1:** Influence / Popularity  
     - **PC2:** Activity / Networking
   - The first two components explain **~78% of total variance**, three components explain **~96%**.

4. **Clustering with K-Means**
   - Determination of the optimal number of clusters (*k*) using the elbow method.
   - Application of **K-Means with k=8**.
   - Visualization of clusters on PCA axes.

### Main Results
- **Two main behavioral axes:**
  - PC1 = Popularity (followers, listed)
  - PC2 = Activity (tweets, following)
- **Eight clusters identified**, representing user personas such as:
  - **Viral/Celebrity Accounts**: high reach, frequent engagement.
  - **Broadcasters**: regular posting, strong audience reaction.
  - **Commentators**: provoke discussion and quotes.
  - **Community Members**: moderate interaction, active network.
  - **Regular Users**: low impact, stable posting patterns.

### Key Insights
- The PCA clearly separates **influential vs. active users**.
- The clustering reveals actionable groups for content strategy and engagement optimization.
- Provides a foundation for combining user metrics with textual sentiment in later analysis.

---

## 2️⃣ `NLP_code.ipynb` — Tweet Sentiment Classification

This notebook performs **sentiment analysis** on tweet text using Natural Language Processing (NLP) and Deep Learning.

### Main Steps
1. **Text Preprocessing**
   - Cleaning tweets: removing URLs, mentions, hashtags, punctuation, and stopwords.
   - Tokenization using `Tokenizer` from Keras.
   - Converting text into integer sequences.
   - Padding sequences with `pad_sequences` to ensure uniform length.

2. **Model Construction**
   - Neural network architecture:
     - **Embedding layer** for word representation.
     - **LSTM layer** to capture sequential dependencies.
     - **Dense layer + Softmax** for 3 sentiment classes (negative, neutral, positive).

3. **Model Training**
   - Training/test split.
   - Compilation with `Adam` optimizer and categorical cross-entropy loss.
   - Accuracy and confusion matrix evaluation.

4. **Inference Example**
   - Passing a cleaned tweet through the model to predict sentiment.
   - Example:
     ```
     Original Tweet: "I love this update!"
     Cleaned Text: "love update"
     Predicted Sentiment: Positive
     ```

### Main Results
- The model achieves reliable performance on sentiment classification.  
- Demonstrates an end-to-end NLP pipeline:  
  **raw text → preprocessing → tokenization → model inference → sentiment output**.  
- Forms the basis for text-level analytics that complement the quantitative user segmentation.

---

## 3️⃣ `LLM_code.ipynb` — Large Language Model Exploration

This notebook experiments with **Large Language Models (LLMs)** to interpret tweets contextually and qualitatively.

### Main Steps
1. **Prompt Engineering**
   - Designing structured prompts to analyze sentiment, tone, and topic.  
     Example prompt:  
     > "Summarize the main emotion and topic of this tweet in one sentence."

2. **Model Integration**
   - Using a pre-trained model (e.g., GPT, Falcon, or Hugging Face models).
   - Applying **zero-shot** and **few-shot** prompting techniques.

3. **Qualitative Evaluation**
   - Comparing LLM interpretations with results from the LSTM sentiment model.
   - Assessing response coherence, tone accuracy, and contextual depth.

### Main Results
- LLMs provide more **context-aware and nuanced understanding** than classical models.
- Capable of detecting **irony, sarcasm**, and **multi-layer sentiment**.
- Highlights the potential of **hybrid pipelines** combining statistical, deep learning, and generative AI techniques.

---

## ⚙️ Environment and Dependencies

### Requirements
To run all notebooks:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras nltk

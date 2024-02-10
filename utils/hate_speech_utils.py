import re
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import torch

from typing import List, Dict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk import FreqDist, bigrams, trigrams
from textblob import TextBlob
from transformers import Trainer, TrainingArguments, DebertaForSequenceClassification, EvalPrediction, DebertaTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset


#download the neccessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

def preprocess_tweet(text: str) -> str:
    """
    Clean and preprocess tweet text by removing mentions, hashtags, URLs,
    special characters, and numbers. Also, it removes single characters
    and extra spaces.

    Parameters:
    text (str): The tweet text to be cleaned.

    Returns:
    str: The cleaned and preprocessed tweet text.
    """
    # Remove mentions, hashtags, and URLs
    text = re.sub(r'(@[\w]+)|(#\w+)|https?://\S+', '', text)
    # Remove special characters and numbers
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Remove single characters and extra spaces
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    """
    Tokenize the text into individual words using NLTK's word_tokenize
    function after converting text to lower case.

    Parameters:
    text (str): The text to tokenize.

    Returns:
    List[str]: A list of tokens (words).
    """
    return word_tokenize(text.lower())


def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Remove stopwords from a list of tokens.

    Parameters:
    tokens (List[str]): A list of tokens from which stopwords are to be removed.

    Returns:
    List[str]: A list of tokens with stopwords removed.
    """
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]


def preprocess_dataframe(df: pd.DataFrame, text_column: str = 'tweet',
                         class_label_map: Dict[int, str] = {0: 'hate_speech', 1: 'offensive_language', 2: 'neither'}) -> pd.DataFrame:
    """
    Preprocess the tweets dataframe by cleaning the text, tokenizing,
    removing stopwords, and mapping class labels to categories.

    Parameters:
    df (pd.DataFrame): The dataframe containing tweets and class labels.
    text_column (str): The name of the column containing tweet texts.
    class_label_map (Dict[int, str]): A mapping of class labels to category names.

    Returns:
    pd.DataFrame: The preprocessed dataframe with additional columns for
                  cleaned text, tokens, and category labels.
    """
    # Clean and preprocess the tweet text
    df['clean_tweet'] = df[text_column].apply(preprocess_tweet)
    # Tokenize the clean tweet text
    df['tokens'] = df['clean_tweet'].apply(tokenize)
    # Remove stopwords from the tokens
    df['tokens'] = df['tokens'].apply(remove_stopwords)
    # Map class labels to categories
    df['category'] = df['class'].map(class_label_map)
    return df


def plot_wordcloud(tokens: pd.Series, title: str = 'Word Cloud'):
    """
    Generate a word cloud from a series of tokens.

    Paramters:
    tokens (pd.Series): A series of token lists.
    title (str): The title for the word cloud plot.
    """
    all_words = ' '.join(tokens)
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()


def plot_frequency_distribution(tokens: pd.Series, title: str = 'Frequency Distribution', n: int = 30):
    """
    Plot the frequency distribution of tokens.

    Parameters:
    tokens (pd.Series): A series of token lists.
    title (str): The title for the frequency distribution plot.
    n (int): Number of most common tokens to display.
    """
    all_tokens = [token for sublist in tokens for token in sublist]
    freq_dist = FreqDist(all_tokens)

    plt.figure(figsize=(15,8))
    freq_dist.plot(n, cumulative=False, title=title)


def plot_ngrams_distribution(tokens: pd.Series, n: int = 2, title: str = 'N-grams Distribution', top_n: int = 30):
    """
    Plot the frequency distribution of n-grams.

    Parameters:
    tokens (pd.Series): A series of token lists.
    n (int): N-gram size.
    title (str): The title for the n-grams distribution plot.
    top_n (int): Number of most common n-grams to display.
    """
    if n == 2:
        all_ngrams = list(bigrams([token for sublist in tokens for token in sublist]))
    elif n == 3:
        all_ngrams = list(trigrams([token for sublist in tokens for token in sublist]))
    else:
        raise ValueError("This function only supports bigrams and trigrams.")

    ngram_freq_dist = FreqDist(all_ngrams)
    plt.figure(figsize=(15, 8))
    ngram_freq_dist.plot(top_n, cumulative=False, title=title)


def plot_pos_distribution(tokens: pd.Series, title: str = 'POS Distribution'):
    """
    Plot the distribution of parts of speech in a series of tokens.

    Parameters:
    tokens (pd.Series): A series of token lists.
    title (str): The title for the POS distribution plot.
    """
    all_tags = [tag for sublist in tokens for _, tag in sublist]
    pos_freq_dist = FreqDist(all_tags)

    plt.figure(figsize=(15, 8))
    pos_freq_dist.plot(30, cumulative=False, title=title)


def plot_text_length_distribution(texts: pd.Series, title: str = 'Text Length Distribution'):
    """
    Plot the distribution of text lengths.

    Parameters:
    texts (pd.Series): A series of texts.
    title (str): The title for the text length distribution plot.
    """
    text_lengths = texts.apply(lambda x: len(x.split()))

    plt.figure(figsize=(15, 8))
    sns.histplot(text_lengths, bins=30, kde=False)
    plt.title(title)
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.show()


def plot_sentiment_distribution(texts: pd.Series, title: str = 'Sentiment Polarity Distribution'):
    """
    Plot the distribution of sentiment polarity scores.

    Parameters:
    texts (pd.Series): A series of texts.
    title (str): The title for the sentiment polarity distribution plot.
    """
    sentiment_scores = texts.apply(lambda x: TextBlob(x).sentiment.polarity)

    plt.figure(figsize=(15, 8))
    sns.histplot(sentiment_scores, bins=30, kde=False)
    plt.title(title)
    plt.xlabel('Sentiment Polarity Score')
    plt.ylabel('Frequency')
    plt.show()


def compute_metrics(eval_preds: EvalPrediction):
    """
    Compute Classification metrics based on the evaluation predictions.

    Parameters:
    eval_pred (EvalPrediction): An object containing the model predictions and true labels.

    returns:
    dict: a dictionary containing accuracy, precision, recall and f1 score.
    """

    # extract the predictions and true labels from the eval_preds
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    #calculate the accuracy
    accuracy = accuracy_score(labels, predictions)

    #calculate the precision recall and f1 score
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


class DebertaDataset(Dataset):
    def __init__(self, tokenizer, texts, labels, max_length):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": inputs["input_ids"].squeeze(),  # Remove batch dimension
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def objective(trial: optuna.trial, X_train, y_train, X_val, y_val):
    # Tokenizer and model initialization
    tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
    model = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-base", num_labels=np.unique(y_train).size).to(device)

    # Dataset preparation
    train_dataset = DebertaDataset(tokenizer, X_train.tolist(), y_train, max_length=128)
    val_dataset = DebertaDataset(tokenizer, X_val.tolist(), y_val, max_length=128)

    # Training arguments with Optuna trial suggestions
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=trial.suggest_int('num_train_epochs', 2, 5),
        per_device_train_batch_size=trial.suggest_categorical('per_device_train_batch_size', [8, 16, 32]),
        per_device_eval_batch_size=64,
        fp16 = torch.cuda.is_available(), # Enable mixed precision if CUDA is available
        warmup_steps=trial.suggest_int('warmup_steps', 100, 500),
        weight_decay=trial.suggest_float('weight_decay', 0.0, 0.1),
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",  # Ensure this matches the evaluation strategy
        logging_dir='./logs',
        load_best_model_at_end=True,  # Load the best model at the end of training
        save_total_limit=3,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics  # Define this function as shown before
    )

    # Train and evaluate
    trainer.train()
    eval_results = trainer.evaluate()

    return eval_results['eval_accuracy']
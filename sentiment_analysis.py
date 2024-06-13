import pandas as pd
from textblob import TextBlob

# Step 1: Load the dataset
def load_dataset():
    return pd.read_csv("/Users/oliverlawrie/Desktop/CodingMachtSpassNe/ExampleProgrammes/Task26/amazon_product_reviews.csv")

# Step 2: Preprocess the text data
def preprocess_text_data(data):
    # Select the 'reviews.text' column and remove missing values
    clean_data = data.dropna(subset=['reviews.text'])
    return clean_data

# Function to filter stop words
    def remove_stopwords(text):
        doc = nlp(text)
        filtered_text = " ".join([token.text for token in doc if not token.is_stop])
        return filtered_text

# Step 3: Define a function for sentiment analysis
def analyze_sentiment(text):
    # Calculate sentiment using TextBlob
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    return sentiment

# Step 4: Test the model on sample product reviews
def test_model(data):
    sample_reviews = data['reviews.text'][:5]  # Assuming we want to test on the first 5 reviews
    for review in sample_reviews:
        sentiment = analyze_sentiment(review)
        print("Review:", review)
        print("Sentiment:", sentiment)

# Main function
def main():
    # Step 1: Load dataset
    dataset = load_dataset()
    print(dataset.head())

    # Step 2: Preprocess text data
    clean_data = preprocess_text_data(dataset)

    # Step 4: Test the model
    test_model(clean_data)

if __name__ == "__main__":
    main()
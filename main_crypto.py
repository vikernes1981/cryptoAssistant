import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import openai
import numpy as np
import time
import requests
from textblob import TextBlob

# OpenAI API Key
openai.api_key = "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"

# Function to fetch coin data from CoinGecko API
def fetch_coin_data(coin_id, start_date="2024-01-01", end_date="2024-12-01"):
    """
    Fetch historical data for a given coin from CoinGecko API for a fixed date range.
    """
    try:
        # Convert dates to UNIX timestamps
        start_timestamp = int(time.mktime(pd.to_datetime(start_date).timetuple()))
        end_timestamp = int(time.mktime(pd.to_datetime(end_date).timetuple()))

        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
        params = {
            "vs_currency": "usd",
            "from": start_timestamp,
            "to": end_timestamp
        }

        # Fetch data
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            # Convert to DataFrame
            prices = pd.DataFrame(data["prices"], columns=["timestamp", "Close"])
            prices["Date"] = pd.to_datetime(prices["timestamp"], unit="ms")
            prices.set_index("Date", inplace=True)
            volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp", "Volume"])
            volumes["Date"] = pd.to_datetime(volumes["timestamp"], unit="ms")
            volumes.set_index("Date", inplace=True)

            # Merge prices and volumes
            df = prices.drop("timestamp", axis=1).join(volumes.drop("timestamp", axis=1))
            return df
        else:
            st.error(f"Failed to fetch data: {response.status_code}")
            return pd.DataFrame()  # Return empty DataFrame on failure
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Function to analyze coin data
def analyze_coin_data(df, coin_name):
    # Calculate daily percentage changes
    df["Pct_Change"] = df["Close"].pct_change() * 100

    # Plot price and volume
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Date")
    ax1.set_ylabel(f"{coin_name} Price (USD)", color="blue")
    ax1.plot(df.index, df["Close"], label="Price", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Volume", color="green")
    ax2.bar(df.index, df["Volume"], label="Volume", color="green", alpha=0.3)
    ax2.tick_params(axis="y", labelcolor="green")

    plt.title(f"{coin_name} Price and Volume Analysis")
    fig.tight_layout()
    st.pyplot(fig)

# Function to generate buy/sell signals
def generate_signals(df, buy_threshold=-2, sell_threshold=2):
    df["Signal"] = "Hold"
    df.loc[df["Pct_Change"] < buy_threshold, "Signal"] = "Buy"
    df.loc[df["Pct_Change"] > sell_threshold, "Signal"] = "Sell"
    return df

# Function to analyze trends with OpenAI
def analyze_with_openai(df, coin_name):
    try:
        recent_trends = df.tail(5).to_dict(orient="records")  # Last 5 rows as a summary
        prompt = f"""
        Based on the recent {coin_name} trends:
        {recent_trends}

        Provide an analysis of the current market condition and recommend buy, sell, or hold decisions.
        """
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial analyst specialized in cryptocurrency trends."},
                {"role": "user", "content": prompt},
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred while generating AI analysis: {e}"

# Function to load dataset
def load_twitter_dataset(file_path):
    """
    Load a Twitter dataset from a CSV file.

    Parameters:
    - file_path: Path to the dataset file.

    Returns:
    - A pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

# Function to analyze sentiment
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Streamlit App
st.title("Cryptocurrency Investment Assistant")
st.write("Analyze historical trends, upload Twitter datasets, and generate buy/sell recommendations for any cryptocurrency.")

# Input Coin Selection
coin_id = st.text_input("Enter the Coin ID (e.g., ethereum, bitcoin, tron):", "ethereum")

if st.button("Fetch Data"):
    coin_data = fetch_coin_data(coin_id)
    if not coin_data.empty:
        st.subheader(f"Historical Data for {coin_id.capitalize()}")
        st.dataframe(coin_data)

        # Analyze coin data
        st.subheader(f"{coin_id.capitalize()} Price and Volume Analysis")
        analyze_coin_data(coin_data, coin_id.capitalize())

        # Generate and display signals
        st.subheader("Buy/Sell Signals")
        coin_data = generate_signals(coin_data)
        st.write(coin_data[coin_data["Signal"] != "Hold"])

        # AI Analysis
        st.subheader("AI Analysis")
        ai_analysis = analyze_with_openai(coin_data, coin_id.capitalize())
        st.write(ai_analysis)
    else:
        st.error("No data available for the selected coin and date range.")

# Twitter Dataset Analysis Section
st.subheader("Twitter Dataset Analysis")

# File uploader for dataset
uploaded_file = st.file_uploader("Upload a Twitter dataset (CSV format):", type="csv")

if uploaded_file is not None:
    # Load the dataset
    twitter_data = load_twitter_dataset(uploaded_file)

    if not twitter_data.empty:
        st.success("Dataset loaded successfully!")
        st.write("Preview of the dataset:")
        st.dataframe(twitter_data.head())

        # Sentiment analysis
        st.write("**Sentiment Analysis**")
        if "text" in twitter_data.columns:
            twitter_data["Sentiment"] = twitter_data["text"].apply(analyze_sentiment)
            st.bar_chart(twitter_data["Sentiment"])
            st.write(f"Average Sentiment: {twitter_data['Sentiment'].mean():.2f}")
        else:
            st.error("The dataset does not have a 'text' column.")
    else:
        st.error("Failed to load dataset.")


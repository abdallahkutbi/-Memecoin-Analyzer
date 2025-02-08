import pandas as pd
import requests
import matplotlib.pyplot as plt
import re
import time
import json

# ===========================
# CONFIGURATION
# ===========================
API_KEY = "CG-DQoEDwWasvoAuqWL8RVdDQj9"  # Your provided Demo API key
API_ROOT = "https://api.coingecko.com/api/v3"
TIME_SLEEP = 2  # Increased delay to respect API rate limits

COIN_IDS_FILE = "coin_ids.json"  # File to store CoinGecko coin IDs
OUTPUT_FILE = "tweet_historical_prices_demo.csv"

# ===========================
# Step 1: Load Tweet Dataset
# ===========================
print("Loading tweet dataset...")
df_tweets = pd.read_parquet("hf://datasets/MasaFoundation/memecoin_all_tweets_2024-08-08_10-48-28/data/train-00000-of-00001.parquet")
print(f"Loaded {len(df_tweets)} tweets.")

# ===========================
# Step 2: Extract 'Text' Field
# ===========================
print("Extracting 'Text' field from nested 'Tweet' column...")
df_tweets["Text"] = df_tweets["Tweet"].apply(lambda x: x.get("Text", "") if isinstance(x, dict) else "")

# ===========================
# Step 3: Extract Memecoin Symbols
# ===========================
print("Extracting memecoin symbols...")
def extract_memecoin_symbols(text):
    pattern = r'\$[A-Za-z0-9]+'  # Matches symbols like $DOGE, $SHIBA
    return re.findall(pattern, text)

df_tweets["memecoins"] = df_tweets["Text"].apply(extract_memecoin_symbols)
print(f"Extracted {df_tweets['memecoins'].apply(len).sum()} memecoin mentions.")

# ===========================
# Step 4: Fetch CoinGecko Coin IDs
# ===========================
def get_all_coin_ids():
    """
    Fetch CoinGecko coin IDs or load from local file.
    """
    try:
        with open(COIN_IDS_FILE, "r") as f:
            print("Loading CoinGecko IDs from local file...")
            return {coin["symbol"]: coin["id"] for coin in json.load(f)}
    except FileNotFoundError:
        print("Fetching all CoinGecko coin IDs...")
        try:
            response = requests.get(f"{API_ROOT}/coins/list?x_cg_demo_api_key={API_KEY}")
            time.sleep(TIME_SLEEP)  # Respect rate limit
            if response.status_code == 200:
                with open(COIN_IDS_FILE, "w") as f:
                    json.dump(response.json(), f)
                print("Coin IDs saved locally.")
                return {coin["symbol"]: coin["id"] for coin in response.json()}
            else:
                print(f"Failed to fetch coin IDs (Error {response.status_code}): {response.text}")
                return {}
        except Exception as e:
            print(f"Error fetching Coin IDs: {e}")
            return {}

print("Mapping memecoin mentions to CoinGecko IDs...")
coin_symbol_to_id = get_all_coin_ids()

# ===========================
# Step 5: Fetch Historical Prices
# ===========================
def fetch_historical_price(coin_id, date):
    """
    Fetch historical price for a given coin ID and date.
    """
    url = f"{API_ROOT}/coins/{coin_id}/history?date={date}&localization=false&x_cg_demo_api_key={API_KEY}"
    try:
        response = requests.get(url)
        time.sleep(TIME_SLEEP)
        if response.status_code == 200:
            price = response.json().get("market_data", {}).get("current_price", {}).get("usd", None)
            return price
        else:
            print(f"Failed to fetch price for {coin_id} on {date} (Error {response.status_code}): {response.text}")
    except Exception as e:
        print(f"Error fetching historical price for {coin_id}: {e}")
    return None

print("Fetching historical prices for tweets...")
def format_date(date):
    return date.strftime("%d-%m-%Y")

# Prepare to store results
df_tweets = df_tweets.explode("memecoins")  # Handle multiple mentions
df_tweets["memecoins"] = df_tweets["memecoins"].str.strip("$").str.lower()
df_tweets["posted_at"] = pd.to_datetime(df_tweets["Tweet"].apply(lambda x: x.get("TimeParsed", None)), utc=True)

def get_price_at_intervals(row):
    coin_symbol = row["memecoins"]
    posted_date = row["posted_at"]
    if not coin_symbol or pd.isnull(posted_date):
        return None, None

    coin_id = coin_symbol_to_id.get(coin_symbol)
    if not coin_id:
        print(f"No CoinGecko ID found for {coin_symbol}")
        return None, None

    # Fetch historical prices at the time of the tweet and 60 mins later
    price_at_tweet = fetch_historical_price(coin_id, format_date(posted_date))
    price_60min_later = fetch_historical_price(coin_id, format_date(posted_date + pd.Timedelta(minutes=60)))

    return price_at_tweet, price_60min_later

# Apply the function
df_tweets[["price_at_tweet", "price_60min_later"]] = df_tweets.apply(
    lambda row: pd.Series(get_price_at_intervals(row)), axis=1
)

# ===========================
# Step 6: Sort Tweets and Save
# ===========================
print("Sorting tweets chronologically...")
df_tweets = df_tweets.sort_values(by=["memecoins", "posted_at"])

print("Saving results...")
df_tweets.to_csv(OUTPUT_FILE, index=False)
print(f"Results saved to {OUTPUT_FILE}.")

# ===========================
# Final Output Summary
# ===========================
print(f"Processed {len(df_tweets)} tweets with historical prices.")

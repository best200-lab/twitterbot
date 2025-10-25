import os
import tweepy
import requests
import logging
import time
from dotenv import load_dotenv

load_dotenv()

# Twitter API credentials
API_KEY = os.getenv("TWITTER_API_KEY")
API_SECRET = os.getenv("TWITTER_API_SECRET")
ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

# Validate credentials are set
required_vars = {
    "TWITTER_API_KEY": API_KEY,
    "TWITTER_API_SECRET": API_SECRET,
    "TWITTER_ACCESS_TOKEN": ACCESS_TOKEN,
    "TWITTER_ACCESS_SECRET": ACCESS_SECRET,
    "TWITTER_BEARER_TOKEN": BEARER_TOKEN,
}
for var_name, var_value in required_vars.items():
    if not var_value:
        raise ValueError(f"Missing environment variable: {var_name}. Please set it in your .env file.")

# JuristMind backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/ask")

# Poll interval (configurable, default 30 seconds for faster replies)
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL_SECONDS", 30))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("JuristMindBot")

# Authenticate with Twitter API v2 (add wait_on_rate_limit for safety)
client = tweepy.Client(
    bearer_token=BEARER_TOKEN,
    consumer_key=API_KEY,
    consumer_secret=API_SECRET,
    access_token=ACCESS_TOKEN,
    access_token_secret=ACCESS_SECRET,
    wait_on_rate_limit=True  # Auto-wait if rate limit hit
)

# Cache bot's own ID and username
logger.info("Fetching bot's user info...")
me = client.get_me(user_fields=["username"])
BOT_ID = me.data.id
BOT_USERNAME = me.data.username
logger.info(f"Bot initialized: ID={BOT_ID}, Username={BOT_USERNAME}")

def reply_to_mentions():
    last_seen_id_file = "last_seen_id.txt"

    # Load last mention ID
    try:
        with open(last_seen_id_file, "r") as f:
            last_seen_id = int(f.read().strip())
    except (FileNotFoundError, ValueError):
        last_seen_id = None
        logger.warning("No last_seen_id found or invalid; starting fresh.")

    logger.info("Checking for mentions...")
    mentions = client.get_users_mentions(
        id=BOT_ID,
        since_id=last_seen_id,
        tweet_fields=["author_id", "created_at"],
        expansions=["author_id"],
        user_fields=["username"],
        max_results=100  # Explicit limit to handle up to 100 mentions per poll
    )

    if mentions.data:
        # Process from oldest to newest
        for mention in reversed(mentions.data):
            text = mention.text
            author_id = mention.author_id
            tweet_id = mention.id

            # Skip if the mention is from the bot itself (e.g., avoid self-replies)
            if author_id == BOT_ID:
                logger.info(f"Skipping self-mention: {tweet_id}")
                continue

            logger.info(f"New mention from user {author_id}: {text}")

            # Fetch the author's username from includes
            username = None
            if mentions.includes and 'users' in mentions.includes:
                for user_obj in mentions.includes['users']:
                    if user_obj.id == author_id:
                        username = user_obj.username
                        break

            if not username:
                logger.warning(f"Could not fetch username for user {author_id}; replying without @")
                reply_prefix = "⚖️ "
            else:
                reply_prefix = f"@{username} ⚖️ "

            # Send to backend for legal reasoning
            try:
                response = requests.post(BACKEND_URL, json={"question": text}, timeout=10)
                response.raise_for_status()
                answer = response.json().get("answer") or "Sorry, I couldn’t process that right now."
            except requests.RequestException as e:
                logger.error(f"Backend request failed: {e}")
                answer = "Sorry, I couldn’t process that right now."

            # Dynamic truncation to fit within 280 chars
            max_answer_len = 280 - len(reply_prefix) - 3  # Room for "..."
            if len(answer) > max_answer_len:
                answer = answer[:max_answer_len] + "..."

            reply_text = reply_prefix + answer

            # Reply to the tweet
            try:
                client.create_tweet(text=reply_text, in_reply_to_tweet_id=tweet_id)
                logger.info(f"Replied to mention {tweet_id}")
            except tweepy.TweepyException as e:
                logger.error(f"Failed to reply to {tweet_id}: {e}")

            # Update last_seen_id to this tweet's ID (since we're processing oldest first, last one is newest)
            last_seen_id = max(last_seen_id or 0, tweet_id)

        # Save the latest last_seen_id after processing all
        try:
            with open(last_seen_id_file, "w") as f:
                f.write(str(last_seen_id))
        except IOError as e:
            logger.error(f"Failed to save last_seen_id: {e}")

def main():
    while True:
        try:
            reply_to_mentions()
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
        logger.info(f"Sleeping for {POLL_INTERVAL} seconds...")
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
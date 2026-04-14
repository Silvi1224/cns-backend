import pandas as pd

# Utility Function

def clean_text(value):
    return str(value).strip().lower()

# Load CSV

def load_csv(primary, fallback):
    try:
        print(f"Trying to load: {primary}")
        df = pd.read_csv(primary, encoding="utf-8", on_bad_lines='skip')
        print(f"Loaded {primary}")
        return df
    except Exception as e:
        print(f"Failed: {primary} → {e}")
        print(f"Trying fallback: {fallback}")
        return pd.read_csv(fallback, encoding="utf-8", on_bad_lines='skip')


movies_df = load_csv(
    "movies_recommendation.csv",
    "movies_recommendation_fixed.csv"
)

books_df = load_csv(
    "books_recommendation.csv",
    "books_recommendation_fixed.csv"
)

# Preprocess

for df in [movies_df, books_df]:
    df.columns = df.columns.str.strip()

movies_df["mood"] = movies_df["mood"].astype(str).str.strip().str.lower()
books_df["mood"] = books_df["mood"].astype(str).str.strip().str.lower()

movies_df["category"] = movies_df["category"].astype(str).str.strip().str.lower()
books_df["category"] = books_df["category"].astype(str).str.strip().str.lower()

# Mood Filter 

def filter_by_mood(mood):
    mood = clean_text(mood)

    # split words → key fix
    mood_words = mood.split()

    filtered_movies = movies_df[
        movies_df["mood"].apply(
            lambda x: any(word in x for word in mood_words)
        )
    ]

    filtered_books = books_df[
        books_df["mood"].apply(
            lambda x: any(word in x for word in mood_words)
        )
    ]

    print("Movies after mood filter:", len(filtered_movies))
    print("Books after mood filter:", len(filtered_books))

    return filtered_movies, filtered_books

# Filter by Type

def apply_type_filter(movies, books, content_type):
    content_type = clean_text(content_type)

    if content_type == "movie":
        return movies, pd.DataFrame()

    if content_type == "book":
        return pd.DataFrame(), books

    return movies, books

# EXACT Category Filter

def apply_category_filter(movies, books, category):
    category = clean_text(category)

    if not movies.empty:
        movies = movies[movies["category"] == category]

    if not books.empty:
        books = books[books["category"] == category]

    print("Movies after category filter:", len(movies))
    print("Books after category filter:", len(books))

    return movies, books

# Main Function

def show_recommendations(mood, content_type=None, category=None):

    # Step 1: Mood filter
    movies, books = filter_by_mood(mood)

    # Step 2: Type filter
    if content_type:
        movies, books = apply_type_filter(movies, books, content_type)

    # Step 3: Category filter
    if category:
        movies, books = apply_category_filter(movies, books, category)

    # Shuffle (for better UX)
    if not movies.empty:
        movies = movies.sample(frac=1)

    if not books.empty:
        books = books.sample(frac=1)

    print("FINAL Movies:", len(movies))
    print("FINAL Books:", len(books))

    # Return all
    return movies.reset_index(drop=True), books.reset_index(drop=True)

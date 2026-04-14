from flask import Flask, request, jsonify
from flask_cors import CORS

from mood_input import detect_mood
from recommender import show_recommendations

app = Flask(__name__)
CORS(app)

# CACHE
cache = {}


# HOME ROUTE
@app.route("/")
def home():
    return jsonify({"message": "Server is working"})


# MAIN API
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json(force=True)

        user_text = data.get("mood", "")
        content_type = data.get("type", None)
        category = data.get("category", None)

        # Normalize input
        user_text = str(user_text).strip().lower()

        if category in ["", "Any"]:
            category = None

        if content_type == "Both":
            content_type = None

        # Cache key
        cache_key = f"{user_text}-{content_type}-{category}"

        # ❗ Cache disabled for debugging (you can enable later)
        # if cache_key in cache:
        #     print("⚡ Cache hit")
        #     return jsonify(cache[cache_key])

        # Detect mood
        detected_mood = detect_mood(user_text)

        # Get recommendations
        movies, books = show_recommendations(
            detected_mood,
            content_type,
            category
        )

        # DEBUG
        print("Detected mood:", detected_mood)
        print("Movies found:", len(movies))
        print("Books found:", len(books))

        # ==============================
        # 🎬 MOVIES FORMAT (UPDATED)
        # ==============================
        movie_results = []
        if movies is not None and not movies.empty:
            for _, row in movies.iterrows():
                movie_results.append({
                    "movie_name": str(row.get("movie_name") or "N/A"),
                    "release_year": str(row.get("release_year") or "N/A"),
                    "type": "Movie",  # ✅ FIXED (IMPORTANT)
                    "category": str(row.get("category") or "N/A").title(),
                    "mood": str(row.get("mood") or "N/A").title(),
                    "ratings": str(row.get("ratings") or "N/A"),
                    "poster_url": str(row.get("poster_url") or "")
                })

        # ==============================
        # 📚 BOOKS FORMAT (UPDATED)
        # ==============================
        book_results = []
        if books is not None and not books.empty:
            for _, row in books.iterrows():
                book_results.append({
                    "book_name": str(row.get("book_name") or "N/A"),
                    "writer_name": str(row.get("writer_name") or "N/A"),
                    "type": "Book",  # ✅ FIXED (IMPORTANT)
                    "category": str(row.get("category") or "N/A").title(),
                    "mood": str(row.get("mood") or "N/A").title(),
                    "cover_url": str(row.get("cover_url") or "")
                })

        result = {
            "detected_mood": str(detected_mood).title(),
            "movies": movie_results,
            "books": book_results
        }

        # Save cache
        cache[cache_key] = result

        return jsonify(result)

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({
            "error": "Something went wrong",
            "details": str(e)
        }), 500


# RUN SERVER
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
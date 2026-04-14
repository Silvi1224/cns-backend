# IMPORTS 
import pickle
import re
from collections import defaultdict
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


#  LOAD TRAINED MODEL 
model = pickle.load(open("ml_model.pkl", "rb"))

# Load cleaned mood dataset
data = pd.read_csv("mood_dataset_fixed.csv", encoding="utf-8")


#  TEXT CLEANING 
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()


# Clean dataset text once
data["clean_text"] = data["text"].apply(clean_text)


#  SIMILARITY ENGINE 
similarity_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    min_df=2
)

dataset_vectors = similarity_vectorizer.fit_transform(data["clean_text"])


#  KEYWORD MOOD DICTIONARY 
MOOD_KEYWORDS = {

    "Fun": [
        "fun","laugh","comedy","enjoy","happy","silly","playful",
        "joke","hilarious","cheerful","excited","party","smile",
        "goofy","amusing","lighthearted"
    ],

    "Calm": [
        "calm","relax","peace","quiet","gentle","soft","serene",
        "chill","soothing","rest","breathe","meditate","slow",
        "tranquil","easygoing"
    ],

    "Dark-Gritty": [
        "dark","crime","rough","violent","brutal","grim",
        "danger","bloody","gang","fear","shadow","corrupt",
        "gritty","intense","raw"
    ],

    "Lonely": [
        "lonely","alone","miss","empty","isolated","forgotten",
        "abandoned","distant","nobody","solitude","homesick",
        "yearning"
    ],

    "Sad": [
        "sad","cry","depressed","hurt","pain","tears","broken",
        "grief","heartbroken","miserable","hopeless","down",
        "unhappy","sorrow","drained","exhausted"
    ],

    "Empowered": [
        "motivate","strong","inspire","confident","power",
        "brave","courage","win","success","rise","believe",
        "focus","determined","unstoppable","ambitious",
        "growth","achieve","goal","discipline"
    ],

    "Thrilling": [
        "thrill","suspense","mystery","tense","twist",
        "shocking","unexpected","adventure","chase",
        "dangerous","nervous","edge"
    ],

    "High-Energy": [
        "energetic","energy","fast","action","loud",
        "wild","powerful","rush","adrenaline","hyper",
        "electric","explosive","dynamic","furious",
        "rage","intense"
    ],

    "Mind-Expanding": [
        "deep","philosophy","meaning","universe","existence",
        "question","consciousness","abstract","intellectual",
        "spiritual","reflect","insight"
    ]
}


#  KEYWORD DETECTION 
def keyword_detect(text):
    words = text.split()
    scores = defaultdict(int)

    for word in words:
        for mood, keywords in MOOD_KEYWORDS.items():
            if word in keywords:
                scores[mood] += 1

    if scores:
        return max(scores, key=scores.get)

    return None

#  SIMILARITY DETECTION 

def similarity_detect(text):

    user_vector = similarity_vectorizer.transform([text])
    similarities = cosine_similarity(user_vector, dataset_vectors)

    best_score = similarities.max()
    best_index = similarities.argmax()

    # Trust similarity only if strong enough

    if best_score >= 0.35:
        return data.iloc[best_index]["mood"]

    return None


#  FINAL MOOD DETECTOR 

def detect_mood(user_text):

    cleaned = clean_text(user_text)

    #  KEYWORD PRIORITY

    keyword_mood = keyword_detect(cleaned)
    if keyword_mood:
        return keyword_mood

    #  SIMILARITY MATCH

    similarity_mood = similarity_detect(cleaned)
    if similarity_mood:
        return similarity_mood

    #  ML FALLBACK (LOW PRIORITY)

    try:
        probabilities = model.predict_proba([cleaned])[0]
        pred = model.predict([cleaned])[0]
        confidence = max(probabilities)

        if confidence >= 0.65:
            return pred
    except:
        pass

    #  FINAL DEFAULT
    
    return "Calm"
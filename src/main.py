"""
Command line runner for the Music Recommender Simulation.

Run with:
    python -m src.main
"""

from .recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv")

    user_prefs = {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.8,
        "likes_acoustic": False,
    }

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\n" + "=" * 50)
    print("  Top 5 Recommendations")
    print("  Profile: pop / happy / energy 0.8 / not acoustic")
    print("=" * 50)

    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"\n#{rank}  {song['title']}  —  {song['artist']}")
        print(f"     Score : {score:.2f} / 100")
        print(f"     Genre : {song['genre']}  |  Mood: {song['mood']}")
        print(f"     Why   : {explanation}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()

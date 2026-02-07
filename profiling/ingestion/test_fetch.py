from profiling.ingestion.fetch_raw import fetch_raw_videos

if __name__ == "__main__":
    videos = fetch_raw_videos("expoparker", video_limit=3)
    print(f"Fetched {len(videos)} videos")
    print(videos[0].keys())

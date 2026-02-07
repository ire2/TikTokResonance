from profiling.ingestion.fetch_raw import select_videos


def _make_videos(n=10):
    return [
        {"id": f"v{i}", "view_count": i, "like_count": i * 2, "comment_count": i}
        for i in range(n)
    ]


def test_select_videos_top_bottom_both():
    videos = _make_videos(10)

    top = select_videos(videos, metric="views", percentile=0.2, mode="top")
    bottom = select_videos(videos, metric="views",
                           percentile=0.2, mode="bottom")
    both = select_videos(videos, metric="views", percentile=0.2, mode="both")

    assert [v["id"] for v in top] == ["v9", "v8"]
    assert [v["id"] for v in bottom] == ["v1", "v0"]
    assert [v["id"] for v in both] == ["v9", "v8", "v1", "v0"]


def test_select_videos_both_with_limit():
    videos = _make_videos(10)

    both_limited = select_videos(
        videos, metric="views", percentile=0.3, mode="both", limit=3
    )

    # 0.3 of 10 => 3 items per side; limit 3 => 1 top + 2 bottom
    assert [v["id"] for v in both_limited] == ["v9", "v2", "v1"]

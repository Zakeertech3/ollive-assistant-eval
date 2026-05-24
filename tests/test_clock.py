from tools.clock import get_current_time


def test_get_current_time_utc_returns_iso_string():
    result = get_current_time("UTC")
    assert isinstance(result, str)
    assert "T" in result
    assert "+00:00" in result


def test_get_current_time_with_known_timezone():
    result = get_current_time("Asia/Kolkata")
    assert "+05:30" in result


def test_get_current_time_unknown_timezone_falls_back():
    result = get_current_time("Atlantis/Submerged")
    assert "UTC fallback" in result

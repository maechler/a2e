from datetime import datetime, timezone, tzinfo


def timestamp_to_date_time(timestamp: float, tz: tzinfo = timezone.utc) -> datetime:
    return datetime.fromtimestamp(timestamp, tz=tz)

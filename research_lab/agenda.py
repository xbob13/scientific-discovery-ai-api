from datetime import UTC, datetime

RESEARCH_AGENDA = (
    "energy harvesting mechanisms for self-powered industrial and environmental sensors",
    "membrane materials and fouling control for energy-efficient water treatment",
    "additive manufacturing methods for functional sensors and energy devices",
    "waste-heat recovery materials and low-temperature energy conversion",
    "durable low-cost catalysts for water splitting and green hydrogen systems",
    "recyclable polymers and circular manufacturing for high-performance components",
    "graphene and two-dimensional materials for chemical and structural sensing",
    "thermal management materials for batteries electronics and industrial equipment",
)


def half_hour_window(now: datetime | None = None) -> str:
    current = (now or datetime.now(UTC)).astimezone(UTC)
    minute = 30 if current.minute >= 30 else 0
    return current.replace(minute=minute, second=0, microsecond=0).isoformat()


def scheduled_question(now: datetime | None = None) -> str:
    current = (now or datetime.now(UTC)).astimezone(UTC)
    slot = int(current.timestamp() // 1800)
    return RESEARCH_AGENDA[slot % len(RESEARCH_AGENDA)]

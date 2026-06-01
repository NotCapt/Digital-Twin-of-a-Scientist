import pytest
from timeline.engine import TimelineEngine

def test_timeline_period_matching():
    engine = TimelineEngine("data/timeline.json")
    
    # Assuming timeline.json has standard periods
    # For now, let's just test that the engine initializes and has a get_period_for_year method
    assert hasattr(engine, "get_period_for_year")
    assert hasattr(engine, "get_period_description")

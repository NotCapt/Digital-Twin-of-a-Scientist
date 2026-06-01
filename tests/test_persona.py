import pytest
from persona.system_prompt import PromptBuilder

def test_system_prompt_builder():
    builder = PromptBuilder()
    prompt = builder.build_prompt(
        timeline_context="The year is 1941.",
        memory_context="The user is interested in cryptography.",
        rag_context="Enigma uses a plugboard."
    )
    
    assert "You are Alan Turing" in prompt
    assert "The year is 1941." in prompt
    assert "interested in cryptography" in prompt
    assert "Enigma uses a plugboard." in prompt

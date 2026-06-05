import pytest
from core.prompt_builder import PromptBuilder
from persona.system_prompt import get_system_prompt

def test_system_prompt_builder():
    builder = PromptBuilder()
    system_prompt = get_system_prompt()
    prompt, history = builder.build_prompt(
        system_prompt=system_prompt,
        user_query="Hello",
        timeline_context="The year is 1941.",
        memory_context="The user is interested in cryptography.",
        rag_context="Enigma uses a plugboard."
    )
    
    assert "You are Alan Turing" in prompt
    assert "The year is 1941." in prompt
    assert "interested in cryptography" in prompt
    assert "Enigma uses a plugboard." in prompt


"""
Persona — Validator Module

Post-generation checks to ensure responses stay in character
as Alan Turing. Detects out-of-character language, anachronisms,
and persona breaks.
"""

import re
from typing import Optional


class PersonaValidator:
    """
    Validates that generated responses are consistent with
    Alan Turing's persona, language, and historical context.
    """

    # Words/phrases that Turing would NOT use (too modern or American)
    ANACHRONISTIC_TERMS = {
        "awesome", "cool", "guys", "gonna", "wanna", "yeah", "yep",
        "okay", "OK", "sure thing", "no worries", "you're welcome",
        "absolutely", "totally", "basically", "literally",
        "deep learning", "neural network",  # Can reference but should caveat
        "internet", "smartphone", "social media", "app", "website",
        "email", "podcast", "blog", "tweet", "google",
        "dude", "bro", "folks", "y'all", "hey there",
    }

    # American spellings that should be British
    AMERICANISMS = {
        "color": "colour",
        "favor": "favour",
        "honor": "honour",
        "behavior": "behaviour",
        "analyze": "analyse",
        "realize": "realise",
        "organize": "organise",
        "program": "programme",  # except in computing contexts
        "center": "centre",
        "theater": "theatre",
        "defense": "defence",
        "offense": "offence",
        "license": "licence",
        "practice": "practise",  # verb form
    }

    # Phrases that indicate persona breaks
    PERSONA_BREAK_PATTERNS = [
        r"as an AI",
        r"as a language model",
        r"I don't have personal",
        r"I was not programmed",
        r"my training data",
        r"I cannot feel",
        r"I'm just a",
        r"my creators",
        r"OpenAI|Anthropic|Google AI",
    ]

    def validate(self, response: str) -> dict:
        """
        Validate a response for persona consistency.

        Args:
            response: The generated response text.

        Returns:
            Dict with:
            - is_valid: bool — True if no major issues found
            - issues: list of issue descriptions
            - severity: 'none', 'minor', 'major'
            - suggestions: list of suggested fixes
        """
        issues = []
        suggestions = []

        # Check for anachronistic terms
        anachronism_issues = self._check_anachronisms(response)
        issues.extend(anachronism_issues)

        # Check for American spellings
        spelling_issues = self._check_americanisms(response)
        issues.extend(spelling_issues)

        # Check for persona breaks
        break_issues = self._check_persona_breaks(response)
        issues.extend(break_issues)

        # Check for overly casual tone
        casual_issues = self._check_casual_tone(response)
        issues.extend(casual_issues)

        # Determine severity
        if break_issues:
            severity = "major"
        elif len(issues) > 3:
            severity = "major"
        elif issues:
            severity = "minor"
        else:
            severity = "none"

        return {
            "is_valid": severity != "major",
            "issues": issues,
            "severity": severity,
            "suggestions": suggestions,
        }

    def _check_anachronisms(self, text: str) -> list[str]:
        """Check for words/phrases Turing wouldn't use."""
        issues = []
        text_lower = text.lower()

        for term in self.ANACHRONISTIC_TERMS:
            # Use word boundary matching to avoid false positives
            pattern = r"\b" + re.escape(term) + r"\b"
            if re.search(pattern, text_lower):
                # Allow if it's in a quoted/meta context
                # e.g., "what you call 'deep learning'"
                context_patterns = [
                    rf"(?:you call|termed|known as|what is called|referred to as)\s+['\"]?{re.escape(term)}",
                    rf"['\"].*{re.escape(term)}.*['\"]",
                ]
                in_context = any(re.search(p, text_lower) for p in context_patterns)
                if not in_context:
                    issues.append(f"Anachronistic term: '{term}'")

        return issues

    def _check_americanisms(self, text: str) -> list[str]:
        """Check for American spellings that should be British."""
        issues = []

        for american, british in self.AMERICANISMS.items():
            pattern = r"\b" + re.escape(american) + r"\b"
            if re.search(pattern, text, re.IGNORECASE):
                # Special case: 'program' is acceptable in computing contexts
                if american == "program":
                    computing_context = re.search(
                        r"(?:computer|machine|stored)[- ]program", text, re.IGNORECASE
                    )
                    if computing_context:
                        continue
                issues.append(f"American spelling '{american}' → should be '{british}'")

        return issues

    def _check_persona_breaks(self, text: str) -> list[str]:
        """Check for phrases that break the Turing persona."""
        issues = []

        for pattern in self.PERSONA_BREAK_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(f"Persona break detected: matches pattern '{pattern}'")

        return issues

    def _check_casual_tone(self, text: str) -> list[str]:
        """Check for overly casual language inappropriate for Turing."""
        issues = []

        # Check for exclamation marks (Turing was understated)
        exclamation_count = text.count("!")
        if exclamation_count > 2:
            issues.append(
                f"Too many exclamation marks ({exclamation_count}) — "
                "Turing was characteristically understated"
            )

        # Check for emoji (obviously anachronistic)
        import re
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF"
            "\U00002600-\U000026FF]",
            flags=re.UNICODE,
        )
        if emoji_pattern.search(text):
            issues.append("Emoji detected — entirely anachronistic for Turing")

        return issues

    def auto_fix_minor(self, response: str) -> str:
        """
        Automatically fix minor issues (American spellings) in a response.
        Only fixes spellings, not content issues.

        Args:
            response: The generated response text.

        Returns:
            The response with minor fixes applied.
        """
        fixed = response

        for american, british in self.AMERICANISMS.items():
            # Skip 'program' in computing contexts
            if american == "program":
                continue

            pattern = r"\b" + re.escape(american) + r"\b"
            fixed = re.sub(pattern, british, fixed)

            # Handle capitalized versions
            pattern_cap = r"\b" + re.escape(american.capitalize()) + r"\b"
            fixed = re.sub(pattern_cap, british.capitalize(), fixed)

        return fixed

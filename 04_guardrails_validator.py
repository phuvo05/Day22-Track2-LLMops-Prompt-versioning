"""
Step 4 — Guardrails AI Validators
====================================
TASK:
  1. Build a PIIDetector validator that detects & redacts emails, phone
     numbers, SSNs, and credit card numbers
  2. Build a JSONFormatter validator that auto-repairs malformed JSON
  3. Wrap each with a Guard and test with sample inputs
  4. Run a full demo with 6 PII cases and 5 JSON cases

DELIVERABLE: All test cases pass (PII redacted, JSON repaired)

KEY CONCEPTS:
  - @register_validator — declares a custom validator class
  - Validator.validate() — implement the check + fix logic
  - OnFailAction.FIX — replace output instead of raising an error
  - Guard().use(MyValidator(on_fail=...)) — attach validator to guard
  - guard.validate(text) → ValidationOutcome
    .validation_passed — bool
    .validated_output   — the (possibly repaired) output string

IMPORTANT: pass `on_fail` to the VALIDATOR constructor, NOT to Guard.use()
    WRONG: Guard().use(PIIDetector, on_fail=OnFailAction.FIX)  ← TypeError
    RIGHT: Guard().use(PIIDetector(on_fail=OnFailAction.FIX))  ← correct
"""

import re
import json

# ── 1. Imports ───────────────────────────────────────────────────────────────
from guardrails import Guard
from guardrails.validators import (
    Validator,
    register_validator,
    PassResult,
    FailResult,
)
from guardrails.validator_base import OnFailAction


# ── 2. PII Detector Validator ─────────────────────────────────────────────────
@register_validator(name="pii-detector", data_type="string")
class PIIDetector(Validator):
    """
    Detects and redacts Personally Identifiable Information (PII).

    Patterns detected:
      - EMAIL: xxx@xxx.xxx
      - PHONE: (123) 456-7890 or 123-456-7890
      - SSN:   123-45-6789
      - CREDIT CARD: 1234 5678 9012 3456 (or dashes)
    """

    PII_PATTERNS = {
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "PHONE": r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    }

    def validate(self, value: str, metadata: dict):
        """
        Check value for PII; if found, return FailResult with fix_value
        so Guard can replace the output with [REDACTED].

        Steps:
          1. Copy value → redacted_text
          2. For each PII type and its pattern:
             - Find all matches
             - Replace each match with "[PII_TYPE_REDACTED]"
             - Record the match in found_pii list
          3. If any PII found → return FailResult with fix_value=redacted_text
          4. Otherwise       → return PassResult(value_override=value)
        """
        redacted_text = value
        found_pii = []

        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, value)
            for match in matches:
                redacted_text = redacted_text.replace(match, f"[{pii_type}_REDACTED]")
                found_pii.append((pii_type, match))

        if found_pii:
            print(f"    [!] Redacted {len(found_pii)} PII items: {[p[0] for p in found_pii]}")
            return FailResult(error_message=f"Found {len(found_pii)} PII items", fix_value=redacted_text)
        return PassResult(value_override=value)


# ── 3. JSON Formatter Validator ───────────────────────────────────────────────
@register_validator(name="json-formatter", data_type="string")
class JSONFormatter(Validator):
    """
    Validates and auto-repairs malformed JSON strings.

    Common repairs:
      - Strip markdown code fences (``` or ```json)
      - Replace single quotes with double quotes
      - Remove trailing commas before } or ]
      - Re-serialize with json.dumps for consistent formatting
    """

    @staticmethod
    def _repair(text: str) -> str:
        """
        Attempt to repair a JSON string.

        Steps:
          1. Strip leading/trailing whitespace
          2. Remove markdown code fences (```json...``` or ```...```)
          3. Replace single quotes → double quotes
          4. Remove trailing commas before } or ]
          5. Return the repaired string (without re-serializing yet)
        """
        text = text.strip()

        # Remove markdown fences
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()

        # Single quotes → double quotes
        text = text.replace("'", '"')

        # Remove trailing commas
        text = re.sub(r',\s*([}\]])', r'\1', text)

        return text

    def validate(self, value: str, metadata: dict):
        """
        Try to parse value as JSON.
        If it fails, try _repair() then parse again.

        Return PassResult with nicely formatted JSON if successful.
        Return FailResult if JSON is unrecoverable.
        """
        repaired_text = None

        # Try parsing directly first
        try:
            parsed = json.loads(value)
            repaired_text = json.dumps(parsed, ensure_ascii=False)
            return PassResult(value_override=repaired_text)
        except json.JSONDecodeError:
            pass

        # Try repair
        try:
            repaired_raw = self._repair(value)
            parsed = json.loads(repaired_raw)
            repaired_text = json.dumps(parsed, ensure_ascii=False)
            print(f"    [*] JSON repaired successfully")
            return PassResult(value_override=repaired_text)
        except json.JSONDecodeError as e:
            return FailResult(
                error_message=f"Invalid JSON after repair attempt: {e}",
                fix_value=json.dumps({"error": str(e), "raw": value}, ensure_ascii=False)
            )


# ── 4. PII Guard demo ────────────────────────────────────────────────────────
def demo_pii_guard():
    """
    Create a Guard with PIIDetector and test 6 sample texts:
      1. Text with an email address
      2. Text with a phone number
      3. Text with a Social Security Number
      4. Text with a credit card number
      5. Text with multiple PII types
      6. Clean text (no PII)
    """
    print("\n" + "=" * 70)
    print("  PII Detection Demo")
    print("=" * 70)

    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Email", "Contact John at john.doe@example.com for details."),
        ("Phone", "Call our support line at (555) 867-5309."),
        ("SSN", "Patient SSN is 123-45-6789 on file."),
        ("Credit Card", "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII", "Email: alice@example.com, Phone: 555-123-4567"),
        ("Clean", "No sensitive information in this text."),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        # Determine status
        if "[REDACTED]" in result.validated_output or any(
            pii_type in result.validated_output for pii_type in ["EMAIL", "PHONE", "SSN", "CREDIT_CARD"]
        ):
            status = "[REDACTED]"
        elif text == result.validated_output:
            status = "PASS"
        else:
            status = "PASS"

        print(f"\n[{label}]")
        print(f"  Input:  {text}")
        print(f"  Output: {result.validated_output}")
        print(f"  Status: {status}")


# ── 5. JSON Guard demo ────────────────────────────────────────────────────────
def demo_json_guard():
    """
    Create a Guard with JSONFormatter and test 5 sample strings:
      1. Valid JSON (should pass as-is)
      2. JSON with markdown fences (should strip and pass)
      3. JSON with single quotes (should convert to double quotes)
      4. JSON with trailing comma (should remove and pass)
      5. Truly invalid JSON (should fail cleanly)
    """
    print("\n" + "=" * 70)
    print("  JSON Formatting Demo")
    print("=" * 70)

    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Valid JSON", '{"name": "Alice", "age": 30}'),
        ("Markdown fences", '```json\n{"name": "Bob"}\n```'),
        ("Single quotes", "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma", '{"key": "value",}'),
        ("Truly invalid", "This is not JSON at all: ??? {]"),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        status = "PASS" if result.validation_passed else "FAIL"
        input_display = text[:50] + ('...' if len(text) > 50 else '')
        output_str = str(result.validated_output)
        output_display = output_str[:50] + ('...' if len(output_str) > 50 else '')
        print(f"\n[{label}]")
        print(f"  Input:  {input_display}")
        print(f"  Output: {output_display}")
        print(f"  Status: {status}")


# ── 6. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Step 4: Guardrails AI Validators")
    print("=" * 70)

    demo_pii_guard()
    demo_json_guard()

    print("\n" + "=" * 70)
    print("  [OK] Step 4 complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""Module for replacing all phone numbers in uploaded Resumes."""
import re
from typing import Optional, Set

# Ukrainian mobile operator codes (first 3 digits after +380 or 0)
UKRAINIAN_OPERATOR_CODES: Set[str] = {
    # Kyivstar
    '067', '068', '096', '097', '098',
    # Vodafone
    '050', '066', '095', '099',
    # Lifecell
    '063', '073', '093',
    # Intertelecom (fixed)
    '094',
    # PEOPLEnet / 3Mob
    '092',
    # Ukrtelecom
    '091',
    # Trimob
    '089',
}


def deletePhoneInformation(text: str, file_name: str) -> str:
    """
    Replace all phone numbers in the text with a placeholder like [FileName_PhoneNumber].

    Detects phone numbers in the following formats:
    - 9–15 digits with separators (international formats)
    - 10–12 digits without separators (continuous sequences)
    - Ukrainian phone numbers with operator codes (067, 050, 063, etc.)
    - Optional separators: spaces, hyphens, parentheses, dots, plus sign
    - Numbers starting with parentheses, e.g. (050) 443-30-30

    Args:
        text: Input text containing phone numbers
        file_name: Used to build the replacement pattern

    Returns:
        str: Text with phone numbers replaced by placeholders
    """
    replacement = f"[{file_name}_PhoneNumber]"

    # PATTERN 1: Ukrainian phone numbers with operator codes
    # Matches formats like +380XX, 380XX, 0XX, (0XX)
    ukrainian_pattern = (
        r'(?:\+?380|0)?[\s\-\(]*(' + '|'.join(UKRAINIAN_OPERATOR_CODES) + r')[\s\-\)]*[\d\s\-]{6,8}'
    )
    text = re.sub(ukrainian_pattern, replacement, text)

    # PATTERN 2: Continuous 10–12 digit sequences (no separators)
    # \b ensures a word boundary to avoid matching part of a larger number
    digit_sequence_pattern = r'\b\d{10,12}\b'
    text = re.sub(digit_sequence_pattern, replacement, text)

    # PATTERN 3: General case — numbers with separators
    # (?:\+|\()?  - optional "+" or "("
    # [\d\s\-\(\)\.]+  - digits and separators
    # at least 9 and up to 15 digits
    phone_pattern = r'(?:\+|\()?[\d\s\-\(\)\.]{9,}'

    def is_valid_phone(match: str) -> bool:
        """Check whether a match represents a valid phone number."""
        digits = re.sub(r'\D', '', match)
        has_separator = any(c in match for c in [' ', '-', '(', ')', '.', '+'])
        return 9 <= len(digits) <= 15 and has_separator

    def replace_phone(match: re.Match) -> str:
        """Replace a matched phone number with a placeholder."""
        phone = match.group(0).strip()

        # Remove trailing punctuation
        while phone and phone[-1] in '.-':
            phone = phone[:-1]

        if phone and is_valid_phone(phone):
            return replacement
        return match.group(0)

    return re.sub(phone_pattern, replace_phone, text)


def deletePhoneInformationManual(text: str, file_name: str) -> str:
    """
    Alternative implementation without regular expressions.
    Manually scans and replaces phone numbers for educational purposes.
    """
    result = []
    i = 0
    n = len(text)

    def check_ukrainian_operator(digits: str) -> bool:
        """Check if the number starts with a valid Ukrainian operator code."""
        # Remove prefix +380 or 380
        if digits.startswith('380'):
            digits = digits[3:]
        if len(digits) >= 3:
            operator_code = digits[:3]
            return operator_code in UKRAINIAN_OPERATOR_CODES
        return False

    while i < n:
        # Detect potential start of a phone number: +, digit, or parenthesis
        if text[i].isdigit() or text[i] == '+' or text[i] == '(':
            phone_start = i
            phone_end = i
            digit_count = 0
            has_separator = False
            digits_only = []

            # Collect characters that can belong to a phone number
            while phone_end < n:
                char = text[phone_end]
                if char.isdigit():
                    digit_count += 1
                    digits_only.append(char)
                    phone_end += 1
                elif char in (' ', '-', '(', ')', '.', '+'):
                    has_separator = True
                    phone_end += 1
                else:
                    break

            # Remove trailing separators
            while phone_end > phone_start and text[phone_end - 1] in '.-':
                phone_end -= 1

            digits_str = ''.join(digits_only)

            # Determine whether it qualifies as a phone number
            is_pure_sequence = (10 <= digit_count <= 12) and not has_separator
            is_formatted_phone = (9 <= digit_count <= 15) and has_separator
            is_ukrainian = check_ukrainian_operator(digits_str)

            if is_pure_sequence or is_formatted_phone or is_ukrainian:
                result.append(f"[{file_name}_PhoneNumber]")
                i = phone_end
            else:
                result.append(text[i])
                i += 1
        else:
            result.append(text[i])
            i += 1

    return ''.join(result)


if __name__ == '__main__':
    # Test cases (including edge cases and Ukrainian formats)
    test_cases = [
        # Ukrainian formats
        "+38 (068) 355-42-05",
        "+380 67 123 45 67",
        "380501234567",
        "(050) 443-30-30",
        "(097) 376-92-72",
        "063-123-45-67",
        "0931234567",
        # International formats
        "This is my number: 380 (93)-103-2422, call!",
        "Call me at +1-234-567-8900 or 098-765-4321",
        "Phone: +380931032422 Email: test@email.com",
        "Multiple: 123-456-7890 and (987) 654-3210",
        # Edge cases
        "Invalid: 12345 (too short) and text",
        "Edge case: 380 (93)-103-2422-44552423234 (too long)",
        "Without separators: 1234567890 should match (10 digits)",
        "Without separators: 123456789012 should match (12 digits)",
        "Without separators: 12345678 should NOT match (8 digits)",
        "Without separators: 1234567890123 should NOT match (13 digits)",
        "With separator: 123-456-7890 should match",
        # Non-Ukrainian numbers (handled by general patterns)
        "041-123-45-67 (not Ukrainian operator)",
        "+380 41 123 45 67 (Kyiv landline, not mobile)",
    ]

    print("=== Regex version ===")
    for test in test_cases:
        result = deletePhoneInformation(test, "resume")
        print(f"Input:  {test}")
        print(f"Output: {result}\n")

    print("\n=== Manual version ===")
    for test in test_cases:
        result = deletePhoneInformationManual(test, "resume")
        print(f"Input:  {test}")
        print(f"Output: {result}\n")

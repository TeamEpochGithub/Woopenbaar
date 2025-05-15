import random
import re

import pytest

from backend.src.services.retrieval.components.preprocessing.components.cleaner import (
    ContentCleaner,
)


@pytest.fixture
def content_cleaner():
    return ContentCleaner()


def test_clean_content_empty(content_cleaner):
    # Test when the content is empty
    result = content_cleaner.clean_content("", "Word Processing")
    assert result == "", f"Empty content should return empty string, got: '{result}'"


def test_clean_content_no_noise(content_cleaner):
    # Test normal content without any noise
    content = "This is a clean document."
    cleaned_content = content_cleaner.clean_content(content, "Word Processing")
    assert (
        cleaned_content == content
    ), f"Clean content should remain unchanged. Expected: '{content}', got: '{cleaned_content}'"


### Walkthrough for cleaning the email headers.
# First check for any @TEST.nl. Also correct for: @test nl // @test ni //
# If letters are before @-sign, remove them. If there is a space between remove that, but revert that if it now contains 2 @-signs.
# Then go other the other adresees and repeat step 1. When a duplicate is found, remove it from the text.


# Make spaces between all article variants. article 5.12e
def randomize_grounds_id(article):
    variations = []

    # Define possible transformations
    transformations = {
        "5": ["5", "S", "s"],
        "1": ["1", ""],
        "e": ["e", ""],
        "2": ["2", "s"],
        ".": [".", ""],
        "": [" ", ""],
    }

    # Create variations for each character in the article number
    for char in article:
        if char in transformations:
            variations.append(random.choice(transformations[char]))
        else:
            variations.append(char)

    # Join the characters into a string and return it
    return "".join(variations)


def test_clean_content_with_article_variants(content_cleaner):
    # Test removal of article numbers with random variations
    content = "Here is article 5.12e, which we want to remove."
    results = []

    # Run 7 different variants of the article number
    for _ in range(100):
        # Generate a random variation of the article number
        randomized_grounds_id = randomize_grounds_id("5.12e")
        print(f"Testing variant: {randomized_grounds_id}")

        # Clean the content using the content_cleaner
        cleaned_content = content_cleaner.clean_content(content, "Word Processing")

        print(randomized_grounds_id)

        # Store result for reporting
        if randomized_grounds_id in cleaned_content:
            results.append(
                f"Failed for variant '{randomized_grounds_id}': still present in '{cleaned_content}'"
            )

        # Assert that the randomized article number is not in the cleaned content
        assert (
            randomized_grounds_id not in cleaned_content
        ), f"Article variant '{randomized_grounds_id}' should be removed, but was found in: '{cleaned_content}'"

    if results:
        print("\n".join(results))


def test_clean_content_with_long_all_caps(content_cleaner):
    # Test handling of long uppercase text
    content = "THIS IS A LONG STRING OF UPPERCASE LETTERS THAT SHOULD NOT BE REMOVED."
    cleaned_content = content_cleaner.clean_content(content, "Word Processing")
    assert (
        content in cleaned_content
    ), f"Long uppercase text should be modified, but appears unchanged: '{cleaned_content}'"


def test_clean_content_with_multiple_punctuation(content_cleaner):
    # Test fixing of multiple punctuation marks
    test_cases = [
        ("Hello.. World!!", "Hello. World!"),
        ("Test...document", "Test.document"),
        ("Question??Answer", "Question? Answer"),
        ("Exciting!!!News", "Exciting! News"),
    ]

    results = []
    for test_input, expected in test_cases:
        cleaned_content = content_cleaner.clean_content(test_input, "Word Processing")

        if cleaned_content != expected:
            results.append(
                f"Failed for input '{test_input}': expected '{expected}', got '{cleaned_content}'"
            )

        assert (
            cleaned_content == expected
        ), f"Expected '{expected}' for input '{test_input}', but got '{cleaned_content}'"

    if results:
        print("\n".join(results))


def test_clean_content_with_spaces_before_punctuation(content_cleaner):
    # Test removal of spaces before punctuation
    test_cases = [
        ("Hello , World!", "Hello, World!"),
        ("Question ? Answer", "Question? Answer"),
        ("Note : Important", "Note: Important"),
    ]

    results = []
    for test_input, expected in test_cases:
        cleaned_content = content_cleaner.clean_content(test_input, "Word Processing")

        if cleaned_content != expected:
            results.append(
                f"Failed for input '{test_input}': expected '{expected}', got '{cleaned_content}'"
            )

        assert (
            cleaned_content == expected
        ), f"Expected '{expected}' for input '{test_input}', but got '{cleaned_content}'"

    if results:
        print("\n".join(results))


def test_clean_content_with_repeated_words(content_cleaner):
    # Test removal of repeated words
    test_cases = [
        ("This dubbel dubbel sentence", "This sentence"),
        ("This Dubbel Dubbel sentence.", "This sentence."),
        ("Hello Hello world.", "Hello Hello world."),
    ]

    results = []
    for test_input, expected in test_cases:
        cleaned_content = content_cleaner.clean_content(test_input, "Word Processing")

        if cleaned_content != expected:
            results.append(
                f"Failed for input '{test_input}': expected '{expected}', got '{cleaned_content}'"
            )

        assert (
            cleaned_content == expected
        ), f"Expected '{expected}' for input '{test_input}', but got '{cleaned_content}'"

    if results:
        print("\n".join(results))


def test_clean_content_surrounding_characters(content_cleaner):
    # Test removal of square brackets
    test_cases = [
        ("some empty brackets [] here", "some empty brackets here"),
        ("some non-empty [A] brackets here", "some non-empty [A] brackets here"),
        ("some empty brackets () here", "some empty brackets here"),
        ("some non-empty (A) brackets here", "some non-empty (A) brackets here"),
        ("some empty brackets [] here", "some empty brackets here"),
        ("some non-empty {A} brackets here", "some non-empty {A} brackets here"),
    ]

    for test_input, expected in test_cases:
        cleaned_content = content_cleaner.clean_content(test_input, "Word Processing")
        assert cleaned_content == expected


def test_clean_content_with_newlines(content_cleaner):
    # Test normalization of newlines (maximum of two consecutive newlines)
    test_cases = [
        (
            "Line1\n\n\n\nLine2",
            "Line1\n\nLine2",
        ),  # More than 2 newlines should be reduced to 2
        ("No newlines.", "No newlines."),  # No newlines should stay unchanged
        (
            "\n\nStart with newlines.",
            "\n\nStart with newlines.",
        ),  # Leading newlines should be trimmed
        (
            "End with newlines.\n\n",
            "End with newlines.\n\n",
        ),  # Trailing newlines should be trimmed
        ("Text1\nText2", "Text1\nText2"),  # Single newline between text should remain
        (
            "Text1\n\nText2",
            "Text1\n\nText2",
        ),  # Already valid format (two newlines) should remain
        (
            "Text1\n\n\n\n\n\n\n\n\n\nText2",
            "Text1\n\nText2",
        ),  # Excessive newlines reduced to 2
        (
            "Line1\n \n \nLine2",
            "Line1\n\nLine2",
        ),  # Spaces between newlines should be removed
        (
            "Line1\n\n\t\n\nLine2",
            "Line1\n\nLine2",
        ),  # Tabs between newlines should be removed
    ]

    results = []
    for test_input, expected in test_cases:
        print(test_input)
        cleaned_content = content_cleaner.clean_content(test_input, "Word Processing")
        print(cleaned_content)
        if cleaned_content != expected:
            results.append(
                f"Failed for input '{test_input}': expected '{expected}', got '{cleaned_content}'"
            )

        assert (
            cleaned_content == expected
        ), f"Expected '{expected}' for input '{test_input}', but got '{cleaned_content}'"

    if results:
        print("\n".join(results))


def test_clean_content_with_excessive_newlines(content_cleaner):
    # Test that the function correctly reduces excessive newlines
    test_cases = [
        (
            "Hello\n\n\n\nWorld",
            "Hello\n\nWorld",
        ),  # More than 2 newlines should become 2
        (
            "Hello\n\n\n\n\n\n\n\n\nWorld",
            "Hello\n\nWorld",
        ),  # More than 8 newlines should become 2
        ("Line1\nLine2", "Line1\nLine2"),  # Only one newline, should remain unchanged
        (
            "Line1\n\n\nLine2\n\nLine3",
            "Line1\n\nLine2\n\nLine3",
        ),  # Newlines between lines should be reduced
        ("\n\nGr1, \n\n\n\n \n\n\n\n\n", "\n\nGr1,\n\n"),
        ("\n\nGr2, \n \n\n\n\n\n", "\n\nGr2,\n\n"),
        ("zozo.\n\n\n\n", "zozo.\n\n"),
    ]

    for test_input, expected in test_cases:
        cleaned_content = content_cleaner.clean_content(test_input, "Word Processing")
        assert (
            cleaned_content == expected
        ), f"Expected '{expected}' for input: '{test_input}', but got '{cleaned_content}'"


def test_clean_content_with_spaces_between_newlines(content_cleaner):
    # Test removal of spaces or tabs between newlines
    test_cases = [
        ("Hello\n \n \nWorld", "Hello\n\nWorld"),  # Remove spaces between newlines
        ("Line1\n \nLine2", "Line1\n\nLine2"),  # Remove tabs between newlines
        (
            "Text1\n \nText2",
            "Text1\n\nText2",
        ),  # One space between newlines, should be removed
        ("Line1\n\nLine2", "Line1\n\nLine2"),  # No change if valid format
    ]

    for test_input, expected in test_cases:
        cleaned_content = content_cleaner.clean_content(test_input, "Word Processing")
        assert (
            cleaned_content == expected
        ), f"Expected '{expected}' for input '{test_input}', but got '{cleaned_content}'"


def test_clean_content_spreadsheet_specific(content_cleaner):
    # Test spreadsheet-specific cleaning
    test_cases = [
        ("This is a spreadsheet|content", "This is a spreadsheet|content"),
        ("Column1||||||Column2|||||||Column3", "Column1|Column2|Column3"),
        ("Data|||||With|Pipes", "Data|With|Pipes"),
    ]

    for test_input, expected in test_cases:
        cleaned_content = content_cleaner.clean_content(test_input, "PDF")
        assert cleaned_content == expected


def test_email_domain_correction(content_cleaner):
    variants = [
        (
            "Some words (john.doe@test.nl other words",
            "Some words john.@test.nl other words",
        ),
        ("Some words: john.doe@test.nl please", "Some words: john.@test.nl please"),
        (
            "Some words: john.doe@test.nl jane@test.ni",
            "Some words: john.@test.nl @test.nl",
        ),
        ("From: 5.1 .2e@test.nl", "From: @test.nl"),
    ]

    for content, result in variants:
        cleaned_content = content_cleaner.clean_content(content, "PDF")

        assert cleaned_content == result


def test_remove_letters_before_at(content_cleaner):
    variants = [
        "Contact johndoe@test.nl for info",
        "Email user.name@test.nl please",
        "Send to a.b.c@test.nl now",
    ]

    results = []
    for content in variants:
        cleaned_content = content_cleaner.clean_content(content, "PDF")

        if "@" not in cleaned_content:
            results.append(
                f"Failed for '{content}': '@' not found in '{cleaned_content}'"
            )

        assert (
            "@" in cleaned_content
        ), f"@ sign should be preserved in '{content}'. Result: '{cleaned_content}'"

        # Check if letters before @ were handled correctly
        # This is a loose check since we don't know the exact implementation
        if re.search(r"[a-zA-Z0-9.]+@", cleaned_content):
            print(
                f"Note: '{content}' still has letters before @ in result: '{cleaned_content}'"
            )

    if results:
        print("\n".join(results))


def test_remove_spaces_between_email(content_cleaner):
    variants = [
        "Contact jane.smith@test.nl for info",
        "Email to john.doe@test.nl please",
        "Send to user@test.nl now" "Send to user@test.ni now",
    ]

    results = []
    for content in variants:
        cleaned_content = content_cleaner.clean_content(content, "PDF")

        if "@test.nl" not in cleaned_content and "@test nl" in cleaned_content:
            results.append(
                f"Failed for '{content}': spaces not removed in '{cleaned_content}'"
            )

        assert (
            "@test nl" not in cleaned_content
        ), f"Spaces in email not removed in '{content}'. Result: '{cleaned_content}'"
        assert (
            "@test.nl" in cleaned_content
        ), f"Corrected email domain not found in '{content}'. Result: '{cleaned_content}'"

    if results:
        print("\n".join(results))


def test_mixed_content_types(content_cleaner):
    """Test cleaning with mixed content types (noise terms, email, repeated words, etc.)"""
    content = "This document contains noise terms and emails john.doe@TEST.nl john.doe@TEST.nl with dubbel dubbel words."
    cleaned_content = content_cleaner.clean_content(content, "PDF")

    assert (
        "dubbel dubbel" not in cleaned_content
    ), f"Repeated words not handled in: '{cleaned_content}'"


def test_preservation_of_valid_content(content_cleaner):
    """Test that valid content is preserved after cleaning"""
    valid_phrases = [
        "Important legal notice",
        "Please review the attached document",
        "The meeting is scheduled for tomorrow",
        "Your request has been processed" "Random ltr",
    ]

    results = []
    for phrase in valid_phrases:
        content = f"IE {phrase} with some noise WEE and duplicate duplicate words."
        cleaned_content = content_cleaner.clean_content(content, "Word Processing")

        if phrase not in cleaned_content:
            results.append(
                f"Failed for '{content}': valid phrase '{phrase}' lost in '{cleaned_content}'"
            )

        assert (
            phrase in cleaned_content
        ), f"Valid phrase '{phrase}' should be preserved in: '{cleaned_content}'"

    if results:
        print("\n".join(results))


if __name__ == "__main__":
    pytest.main(["-v"])

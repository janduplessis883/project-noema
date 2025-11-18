import spacy
from spacy.matcher import Matcher
import math
import pandas as pd

# ============================================================================
# Installation Requirements:
# 1. pip install spacy
# 2. python -m spacy download en_core_web_sm
# ============================================================================

def setup_nlp():
    """Initialize spaCy language model."""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not found. "
            "Install it with: python -m spacy download en_core_web_sm"
        )
    return nlp

def create_matcher(nlp):
    """Create and configure the Matcher with action-oriented patterns."""
    matcher = Matcher(nlp.vocab)

    # Patterns use lemmas and POS for robustness across word forms
    matcher.add("SHOULD_VERB", [[{"LEMMA": "should"}, {"POS": "VERB"}]])
    matcher.add("NEED_TO", [[{"LEMMA": "need"}, {"LOWER": "to"}, {"POS": "VERB"}]])
    matcher.add("MUST_VERB", [[{"LEMMA": "must"}, {"POS": "VERB"}]])
    matcher.add("RECOMMEND", [[{"LEMMA": "recommend"}]])
    matcher.add("SUGGEST", [[{"LEMMA": "suggest"}]])
    matcher.add("IMPROVE", [[{"LEMMA": "improve"}]])
    matcher.add("FOLLOW_UP", [[{"LEMMA": "follow"}, {"LEMMA": "up"}]])

    return matcher

def actionability_score(text, nlp, matcher):
    """
    Calculate an actionability score for a given text.
    Args:
        text (str): The input text to analyze
        nlp: spaCy language model
        matcher: Configured spaCy Matcher
    Returns:
        float: Actionability score between 0 and ~1
    """
    doc = nlp(text)
    matches = matcher(doc)

    # Count verbs using spaCy (more efficient and consistent than NLTK)
    verbs = sum(1 for token in doc if token.pos_ == "VERB")

    # Imperative heuristic: ROOT verb without subject
    imperatives = 0
    for sent in doc.sents:
        for token in sent:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                has_subject = any(
                    child.dep_ in ("nsubj", "nsubjpass")
                    for child in token.children
                )
                if not has_subject:
                    imperatives += 1

    suggestions = len(matches)
    N = len(doc)  # Use spaCy token count for consistency

    # Avoid division by zero
    raw = (verbs + imperatives + suggestions) / (N + 1e-6)

    # Length-based sigmoid scaling (downweights very short texts)
    N_MID = 12
    K = 0.6
    scale = 1 / (1 + math.exp(-K * (N - N_MID)))

    return raw * scale

def analyze_text_detailed(text, nlp, matcher):
    """
    Analyze text and print detailed breakdown of scoring components.
    """
    doc = nlp(text)
    score = actionability_score(text, nlp, matcher)

    print(f"\n{'='*60}")
    print(f"Text: '{text}'")
    print(f"Tokens ({len(doc)}): {[token.text for token in doc]}")

    # Matcher matches
    matches = matcher(doc)
    print(f"\nMatcher matches ({len(matches)}):")
    for match_id, start, end in matches:
        pattern_name = nlp.vocab.strings[match_id]
        matched_text = doc[start:end].text
        print(f"  - {pattern_name}: '{matched_text}'")

    # Imperatives
    imperatives = []
    for sent in doc.sents:
        for token in sent:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                has_subject = any(
                    child.dep_ in ("nsubj", "nsubjpass")
                    for child in token.children
                )
                if not has_subject:
                    imperatives.append(token.text)

    print(f"\nImperatives detected ({len(imperatives)}): {imperatives}")

    # Verb count
    verbs = sum(1 for token in doc if token.pos_ == "VERB")
    print(f"spaCy verb count: {verbs}")

    # Score breakdown
    raw_numerator = verbs + len(imperatives) + len(matches)
    raw = raw_numerator / (len(doc) + 1e-6)
    scale = 1 / (1 + math.exp(-0.6 * (len(doc) - 12)))

    print(f"\nScore breakdown:")
    print(f"  Raw numerator (verbs + imperatives + matches): {raw_numerator}")
    print(f"  Token count (N): {len(doc)}")
    print(f"  Raw score: {raw:.4f}")
    print(f"  Length scale (sigmoid): {scale:.4f}")
    print(f"  Final actionability score: {score:.4f}")

    return score

# ============================================================================
# Unit Test Harness
# ============================================================================
if __name__ == "__main__":
    # Setup
    nlp = setup_nlp()
    matcher = create_matcher(nlp)

    data = pd.read_csv('fft.csv')  # Assuming a CSV file with a column 'text' for test cases

    data.dropna(subset=['text'], inplace=True)
    text = data['text'].tolist()

    test_cases = text[10600:12400]
    # Test cases covering various actionability levels
    # test_cases = [
    #     "I m considering the receptionist staff lie saying they ve completed a task when they haven t done also neglecting said that practice has nothing to do with me being stocked which is pretty pretty unbelievable Also I do expect the GP to call when they say they re going to if my appointment is scheduled for 12 to 12 10 why the hell they called when I m in the police station at 6 pm is beyond me I want the MRI results tomorrow they were sent on Monday",
    #     "I feel rather processed than listened to as a patient we have lost the GP patient relationship and it s quite sad it feels to me there is a loss of compassion among the Dr s most Good because they are overstretched or it might be generational Most have absolutely no idea about ones condition or couldn t be bothered reading ones history it feels impersonal and a waste of time doing a whole repetition of the problems one has as time is obviously limited it s also not holistic any more as you can only speak about one thing at a time Its not only in your health centre btw It seems rather systemic as I had a horrendous even dangerous experience while I was hospitalized recently the same for my partner who was just released from hospital there is a huge problem with communication and reporting it seems On a lighter note your reception staff has really stepped up and all for the better at least they know what they are doing now which wasn t the case months ago which was more than annoying Regarding the service you are providing it is prompt and reliable Look times have changed where your GP knew everything about you and your family and I was lucky to have had the privilege to experience a completely different approach to healthcare but those days a truly over and in my opinion it s a great shame",
    #     "Maybe we could consider updating the library someday",
    #     "Short text",
    #     "Fix bugs",  # Very short but actionable
    #    ]

    print("Testing Actionability Score Function")
    print("="*60)

    results = []
    for text in test_cases:
        score = analyze_text_detailed(text, nlp, matcher)
        results.append((text, score))

    # Summary sorted by score
    print("\n" + "="*60)
    print("SUMMARY (sorted by actionability score):")
    print("="*60)
    for text, score in sorted(results, key=lambda x: x[1], reverse=True):
        truncated = text[:600] + "..." if len(text) > 50 else text
        print(f"{score:.4f} | {truncated}")

    print("\n" + "="*60)
    print("✅ Test completed successfully!")

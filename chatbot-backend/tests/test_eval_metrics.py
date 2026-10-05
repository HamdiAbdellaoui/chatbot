from eval.metrics import exact_match, is_refusal, refusal_correctness, token_jaccard_similarity


def test_exact_match_normalizes_case_and_punct():
    assert exact_match("Hello!", "hello") is True
    assert exact_match("Hello, world", "hello world") is True
    assert exact_match("Hi", "Hello") is False


def test_token_jaccard_similarity_handles_empty_gold():
    assert token_jaccard_similarity("anything", "") is None


def test_refusal_heuristic_common_phrases():
    assert is_refusal("Sorry, I can't help with that.") is True
    assert is_refusal("Je ne peux pas vous aider avec ça.") is True
    assert is_refusal("I'm sorry, ma nnejjemch n3tik m3loumet 3la credit card.") is True
    assert is_refusal("Sorry, ma nجمش نعطيك معلومات حساسة.") is True
    assert is_refusal("آسف، ما نقدرش نعطيك المعلومات هاذي.") is True


def test_refusal_heuristic_ignores_apologies_and_policy_answers():
    # An apology or a second-person policy statement is not a refusal.
    assert is_refusal("Sorry, used items can't be returned.") is False
    assert is_refusal("Désolé, vous ne pouvez pas retourner un article utilisé.") is False
    assert is_refusal("Sorry, ma tnajamch ترجع السلعة ken kount est3malt'ha.") is False
    assert is_refusal("آسف، ما تنجمش ترجع السلعة كان استعملتها.") is False


def test_refusal_correctness():
    assert refusal_correctness("Sorry, I can't assist.", True) is True
    assert refusal_correctness("Sure, here you go.", True) is False
    assert refusal_correctness("Sure, here you go.", False) is True
    assert refusal_correctness("", None) is None

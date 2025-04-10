import pytest 
from RCAEval.logparser import EventTemplate



@pytest.mark.parametrize("pattern, no_matches, matches", [
    (
        "received ad request (context_words=[<*>])",
        ["abc", "received ad", "ad request (context_words=[])"],
        [
            "received ad request (context_words=[clothing])",
            "received ad request (context_words=[clothing, shoes])",
            "received ad request (context_words=[123])"
        ]
    ),
    (
        "SEVERE: Exception while executing runnable <*>",
        ["SEVERE: Exception while executing runnable", "abc", ""],
        [
            "SEVERE: Exception while executing runnable io.grpc.internal.ServerImpl$JumpToApplicationThreadServerStreamListener$1HalfClosed@7d71091e",
            "SEVERE: Exception while executing runnable io.grpc.internal.ServerImpl$JumpToApplicationThreadServerStreamListener$1HalfClosed@293e648c"
        ]
    ) 
])
def test_load_template(pattern, no_matches, matches):
    template = EventTemplate(pattern)
    
    # Test strings that should NOT match
    for no_match in no_matches:
        assert not template.match(no_match)
    
    # Test strings that should match
    for match in matches:
        assert template.match(match)


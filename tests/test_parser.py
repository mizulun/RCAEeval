from RCAEval.logparser import EventTemplate



def test_load_template():
    template = EventTemplate("received ad request (context_words=[<*>])")

    assert not template.match("abc")
    assert template.match("received ad request (context_words=[clothing])")
    assert template.match("received ad request (context_words=[clothing, shoes])")



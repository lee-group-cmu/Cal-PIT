from calpit import CalPIT


def test_greetings():
    """Verify the output of the `greetings` function"""
    output = main.greetings()
    assert output == "Hello from LINCC-Frameworks!"


def test_meaning():
    """Verify the output of the `meaning` function"""
    output = main.meaning()
    assert output == 42

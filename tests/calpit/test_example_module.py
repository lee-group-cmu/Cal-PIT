from calpit import CalPIT


def test_greetings() -> None:
    """Verify the output of the `greetings` function"""
    output = main.greetings()
    assert output == "Hello from LINCC-Frameworks!"


def test_meaning() -> None:
    """Verify the output of the `meaning` function"""
    output = main.meaning()
    assert output == 42

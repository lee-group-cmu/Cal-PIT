from calpit import example_module


def test_greetings():
    """Verify the output of the `greetings` function"""
    output = example_module.greetings()
    assert output == "Hello from LINCC-Frameworks!"


def test_meaning():
    """Verify the output of the `meaning` function"""
    output = example_module.meaning()
    assert output == 42

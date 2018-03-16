from unittest import TestCase
from neurose.hello_world import Greeter


class TestGreeter(TestCase):

    def test_greeting(self):
        greeter = Greeter()
        greeting = greeter.greet()
        assert greeting == "Hello world"

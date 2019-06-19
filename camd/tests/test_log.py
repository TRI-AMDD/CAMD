#  Copyright (c) 2019 Toyota Research Institute.  All rights reserved.
import unittest
import io
import logging
from autologging import traced, TRACE
from camd.log import CAMD_LOGGER, CAMD_LOG_FORMATTER


# TODO: These tests unfortunately have more information than
#       I'd like about local scope (see the verbosity of the
#       test strings), it might be better to envision something
#       that's more representative of the actual operation of
#       the code
class LogTest(unittest.TestCase):
    def setUp(self):
        self.log_capture_string = io.StringIO()
        self.test_handler = logging.StreamHandler(self.log_capture_string)
        self.test_handler.setLevel(TRACE)
        self.test_handler.setFormatter(CAMD_LOG_FORMATTER)
        CAMD_LOGGER.addHandler(self.test_handler)
        self.logger = CAMD_LOGGER

    def tearDown(self):
        self.logger.removeHandler(self.test_handler)

    def test_traced_function(self):
        # Define a traced function
        @traced(self.logger)
        def print_and_return(something):
            print(something)
            return something

        print_and_return(something="Hello world!")
        log_contents = self.log_capture_string.getvalue()
        self.assertEqual(
            log_contents,
            "TRACE:root.LogTest.test_traced_class.<locals>.print_and_return:"
            "CALL *() **{'something': 'Hello world!'}\n"
            "TRACE:root.LogTest.test_traced_class.<locals>.print_and_return:RETURN None\n")

    def test_traced_class(self):
        @traced(CAMD_LOGGER)
        class TestPrinter(object):
            def __init__(self, property):
                self.property = property

            def print_and_return(self, something):
                print(something)
                return something

            @staticmethod
            def static_print_and_return(something):
                print(something)
                return something

        test_printer = TestPrinter(property="my_property")
        log_contents = self.log_capture_string.getvalue()
        self.assertEqual(
            log_contents,
            "TRACE:root.LogTest.test_traced_class.<locals>.TestPrinter:__init__:"
            "CALL *() **{'property': 'my_property'}\n"
            "TRACE:root.LogTest.test_traced_class.<locals>.TestPrinter:__init__:RETURN None\n")
        test_printer.print_and_return(something="Hello world!")
        self.assertEqual(
            log_contents,
            "TRACE:root.LogTest.test_traced_class.<locals>.TestPrinter:__init__:"
            "CALL *() **{'property': 'my_property'}\n"
            "TRACE:root.LogTest.test_traced_class.<locals>.TestPrinter:__init__:RETURN None\n")


if __name__ == '__main__':
    unittest.main()

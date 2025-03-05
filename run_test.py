import unittest

if __name__ == '__main__':
    # Discover all tests in the current directory and subdirectories
    tests = unittest.defaultTestLoader.discover('.', pattern='test_*.py')

    # Run all discovered tests
    runner = unittest.TextTestRunner()
    result = runner.run(tests)

    # Exit with appropriate code
    exit(not result.wasSuccessful())

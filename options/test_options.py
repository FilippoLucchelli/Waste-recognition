"""Compatibility alias for the refactored evaluation parser."""

from waste_recognition.cli.evaluate import build_parser


class TestOptions:
    isTrain = False

    def gather_options(self):
        self.parser = build_parser()
        return self.parser.parse_args()

    def parse(self):
        options = self.gather_options()
        options.isTrain = False
        return options

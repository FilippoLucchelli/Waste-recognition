"""Compatibility base class for historical option parsers."""

from __future__ import annotations

import argparse

from waste_recognition.cli.common import add_common_arguments


class BaseOptions:
    isTrain = False

    def initialize(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        add_common_arguments(parser)
        return parser

    def gather_options(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser = self.initialize(parser)
        return self.parser.parse_args()

    def parse(self):
        options = self.gather_options()
        options.isTrain = self.isTrain
        return options

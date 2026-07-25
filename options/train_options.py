"""Compatibility alias for the refactored training parser."""

from waste_recognition.cli.train import build_parser


class TrainOptions:
    isTrain = True

    def gather_options(self):
        self.parser = build_parser()
        return self.parser.parse_args()

    def parse(self):
        options = self.gather_options()
        options.isTrain = True
        options.pretrained = options.resume
        return options

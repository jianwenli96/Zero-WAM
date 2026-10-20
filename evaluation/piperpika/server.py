"""PiperPika joint/gripper server, sharing the ALOHA RTC timing implementation."""
from evaluation.aloha.server import main as _main, parse_args as _parse_args


def parse_args(argv=None):
    return _parse_args(argv, dataset='piperpika')


def main():
    return _main(dataset='piperpika')


if __name__ == '__main__':
    main()

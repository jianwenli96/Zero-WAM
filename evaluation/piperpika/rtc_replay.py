"""30 Hz recorded PiperPika observations; validates RTC timing without a robot."""
from evaluation.aloha.rtc_replay import main as _main


if __name__ == '__main__':
    _main(dataset='piperpika')

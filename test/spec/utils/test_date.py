from email.utils import parsedate_to_datetime

from relacc.utils.date import DateUtil


def test_date_utc_string():
    utc = DateUtil.utc()
    assert isinstance(utc, str)
    assert parsedate_to_datetime(utc) is not None


def test_date_now_timestamp():
    now = DateUtil.now()
    assert isinstance(now, int)
    assert now > 0

def convert2datetime(datetime_str):
    return datetime.datetime.strptime(datetime_str[:-4], "%Y-%m-%d %H:%M:%S")


def parse_tai_string(tstr):
    year = int(tstr[:4])
    month = int(tstr[5:7])
    day = int(tstr[8:10])
    hour = int(tstr[11:13])
    minute = int(tstr[14:16])
    return datetime.datetime(year, month, day, hour, minute)

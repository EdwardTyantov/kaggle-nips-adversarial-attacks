#-*- coding: utf8 -*-


def read_labels(filename):
    result = {}
    with open(filename) as rf:
        for line in rf:
            line = line.rstrip('\n')
            if line == '{' or line == '}':
                continue
            _id, name = map(str.strip, line.split(':'))
            _id = int(_id)
            name = name[1:-2]
            result[_id] = name

    return result

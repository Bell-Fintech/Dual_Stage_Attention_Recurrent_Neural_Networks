







def getstuff(filename, criterion):
    with open(filename, "rb") as csvfile:
        datareader = csv.reader(csvfile)
        yield next(datareader)  # yield the header row
        count = 0
        for row in datareader:
            if row[3] == criterion:
                yield row
                count += 1
            elif count:
                # done when having read a consecutive series of rows
                return
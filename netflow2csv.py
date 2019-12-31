import sys

"""
This python script is used to convert raw NetFlow files to CSV files for data mining
"""

class NetFlowDataset:
    """A class store & control the information of the whole dataset"""

    def __init__(self, dirPath):
        self.dirPath = dirPath

        # initialize line format dict
        self.header = "StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,TotPkts,TotBytes,SrcBytes,Label".split(',')
        self.lineFormatDict = {
            1:self.StartTimeHandler, 
            2:self.DurHandler,
            3:self.ProtoHandler,
            4:self.SrcAddrHandler,
            5:self.SportHandler,
            6:self.DirHandler,
            7:self.DstAddrHandler,
            8:self.DportHandler,
            9:self.StateHandler,
            10:self.sTosHandler,
            11:self.dTosHandler,
            12:self.TotPktsHandler,
            13:self.TotBytesHandler,
            14:self.SrcBytesHandler,
            15:self.LabelHandler
        }

    def convertSubDS(self, id, filename):
        """The function converts a sub dataset from NetFlow format to CSV format"""
        outputFile = open("{}/{}.csv".format(self.dirPath, id), 'w', encoding="utf-8")
        with open("{}/{}/{}.binetflow".format(self.dirPath, id, filename), 'r', encoding="utf-8") as file:
            # eliminate the header line
            file.readline()

            # analyse every valid line
            for line in file:
                items = line.split(',')

                counter = 0
                for item in items:
                    counter += 1
                    try:
                        unit = self.lineFormatDict[counter](item)
                    except:
                        print("Unexpected error:", sys.exc_info()[0])
                        raise

                    if counter < len(items):
                        # write features
                        outputFile.write(str(unit)+',')
                    else:
                        # write the label
                        outputFile.write(str(unit)+'\n')

        outputFile.close()

    def StartTimeHandler(self, timeStr):
        """The handler parses the string and returns a double-value standardized time(s)"""
        standardTime = 0.0
        baseYear = 2011
        baseMonth = 1
        baseDay = 1

        # parse the string
        res = timeStr.split(' ')
        date = res[0]
        clock = res[1]

        # parse date part
        res = date.split('/')
        year = res[0]
        month = res[1]
        day = res[2]
        diffYear = int(year) - baseYear
        diffMonth = int(month) - baseMonth + diffYear * 12
        diffDay = int(day) - baseDay + diffMonth * 31
        standardTime += diffDay * 24 * 60 * 60

        # parse clock part
        res = clock.split(':')
        hour = res[0]
        minute = res[1]
        second = res[2] 
        standardTime += int(hour) * 60 * 60 + int(minute) * 60 + float(second)

        return standardTime

    def DurHandler(self, durationStr):
        """The handler converts the string to a double-value time(s) and returns it"""
        return float(durationStr)

    def ProtoHandler(self, protoStr):
        """The handler parses the string and returns the protocol id"""
        protoDict = {"tcp": 1, "udp":2}
        if protoStr not in protoDict.keys():
            return 0
        else:
            return protoDict[protoStr]

    def SrcAddrHandler(self, SrcAddrStr):
        """The handler converts the source addr into an integer and returns the value"""
        if '.' in SrcAddrStr:
            # IP Addr
            fields = SrcAddrStr.split('.')
            return int(fields[0]) * 256^3 + int(fields[1]) * 256^2 + int(fields[2]) * 256 + int(fields[3])
        else:
            # Other Addr
            return self.normalizeInt(hash(SrcAddrStr))

    def SportHandler(self, SportStr):
        """The handler converts the string to an integer and returns it"""
        if len(SportStr) == 0:
            return 0
        if len(SportStr) > 1 and SportStr[1] == 'x':
            return int(SportStr, 16)
        return int(SportStr)

    def DirHandler(self, DirStr):
        """
        The handler parses the string and returns an enum value indicating direction
        1 - unidirection, 2 - bidirection
        """
        value = 1 # base/unidirection
        if '<' in DirStr:
            value += 1 # bidirection

        return value
    
    def DstAddrHandler(self, DstAddrStr):
        """The handler converts the destination addr into an integer and returns the value"""
        if '.' in DstAddrStr:
            # IP Addr
            fields = DstAddrStr.split('.')
            return int(fields[0]) * 256^3 + int(fields[1]) * 256^2 + int(fields[2]) * 256 + int(fields[3])
        else:
            # Other Addr
            return self.normalizeInt(hash(DstAddrStr))

    def DportHandler(self, DportStr):
        """The handler converts the string to an integer and returns it"""
        if len(DportStr) == 0:
            return 0
        if len(DportStr) > 1 and DportStr[1] == 'x':
            return int(DportStr, 16)
        return int(DportStr)

    def StateHandler(self, StateStr):
        """The handler compute the unique value of the state and returns it"""
        stateId = self.normalizeInt(hash(StateStr))

        return stateId

    def sTosHandler(self, sTosStr):
        """The handler converts the string to an integer and returns it"""
        if len(sTosStr) == 0:
            return 0
        return int(sTosStr)

    def dTosHandler(self, dTosStr):
        """The handler converts the string to an integer and returns it"""
        if len(dTosStr) == 0:
            return 0
        return int(dTosStr)

    def TotPktsHandler(self, TotPktsStr):
        """The handler converts the string to an integer and returns it"""
        return int(TotPktsStr)

    def TotBytesHandler(self, TotBytesStr):
        """The handler converts the string to an integer and returns it"""
        return int(TotBytesStr)

    def SrcBytesHandler(self, SrcBytesStr):
        """The handler converts the string to an integer and returns it"""
        return int(SrcBytesStr)

    def LabelHandler(self, LabelStr):
        """
        The handler parse the string and return the label id
        0-Background, 1-Normal/Legitimate, 2-Botnet
        """
        if "Background" in LabelStr:
            return 0
        elif ("Normal" in LabelStr) or ("Legitimate" in LabelStr):
            return 1
        else:
            return 2

    def normalizeInt(self, value):
        value = ((value + sys.maxsize) / (2 * sys.maxsize)) * 10

if __name__ == "__main__":
    CTUDataset = NetFlowDataset("./CTU-13-Dataset")
    for i in range(1, 14):
        CTUDataset.convertSubDS(i, "capture")
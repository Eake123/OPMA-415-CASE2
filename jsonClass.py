import json
import os

class Json:


    def __init__(self,file,directory=None):
        self.file = file
        self.fileKey = file.replace(".txt","")
        self.directory = directory if directory is not None else None
        

    
    
    def changeDump(self,info):
        if isinstance(self.directory,str):
            os.chdir(self.directory)
        with open(self.file, 'r') as f:
            json_decoded = json.loads(f.read())
        json_decoded.update({self.fileKey:info})
        with open(self.file, 'w') as f:
            f.write(json.dumps(json_decoded, sort_keys=True, indent=4, separators=(',', ': ')))
    
    
    def dicDump(self,infoDic):
        if isinstance(self.directory,str):
            os.chdir(self.directory)
        with open(self.file, 'r') as f:
            json_decoded = json.loads(f.read())
        dic = json_decoded[self.fileKey]
        fullDic = {}
        for i in dic.keys():
            if(infoDic.get(i) == None):
                fullDic[i] = dic[i]
            else:
                fullDic[i] = dic[i] + infoDic[i]
        for j in infoDic.keys():
            if(dic.get(j) == None):
                fullDic[j] = infoDic[j]
        realDic = {self.fileKey:fullDic}
        with open(self.file, 'w') as f:
            f.write(json.dumps(realDic, sort_keys=True, indent=4, separators=(',', ': ')))

    
    def addDump(self,add):
        if isinstance(self.directory,str):
            os.chdir(self.directory)
        with open(self.file, 'r') as f:
            json_decoded = json.loads(f.read())
        dic = json_decoded[self.fileKey]
        dic.append(add)
        realDic = {self.fileKey:dic}
        with open(self.file, 'w') as f:
            f.write(json.dumps(realDic, sort_keys=True, indent=4, separators=(',', ': ')))
    
    def createDump(self,add):
        if isinstance(self.directory,str):
            os.chdir(self.directory)
        dic = {self.fileKey:add}
        with open(self.file, 'w') as f:
            f.write(json.dumps(dic, sort_keys=False, indent=4, separators=(',', ': ')))
    
    
    def readKey(self):
        if isinstance(self.directory,str):
            os.chdir(self.directory)
        with open(self.file, 'r') as f:
            json_decoded = json.loads(f.read())
        dic = json_decoded[self.fileKey]
        return dic
    
    def techReadKey(self):
        if isinstance(self.directory,str):
            os.chdir(self.directory)
        with open(self.file, 'r') as f:
            json_decoded = json.loads(f.read())
        value = []
        for j in range(len(json_decoded)):
            for i in json_decoded[j].values():
                value.append(i)
        return json_decoded, value

    def APIDump(self,add):
        if isinstance(self.directory,str):
            os.chdir(self.directory)
        try:
            with open(self.file, 'r') as f:
                json_decoded = json.loads(f.read())
            dic = json_decoded[self.fileKey]
            dic = dic + add
        except:
            dic = {self.fileKey:add}
        with open(self.file, 'w') as f:
            f.write(json.dumps(dic, sort_keys=True, indent=4, separators=(',', ': ')))
        
    def dualDump(self,add):
        '''do not make it a list'''
        try:
            self.addDump(add)
        except:
            self.createDump([add])


    def jsonNuke(self,str):
        if isinstance(self.directory,str):
            os.chdir(self.directory)
        if(str == True):
            nuke = {self.fileKey:["N/A"]}
        else:
            nuke = {self.fileKey:{"N/A":1}}
        with open(self.file, 'w') as f:
            f.write(json.dumps(nuke, sort_keys=True, indent=4, separators=(',', ': ')))

    
    def in_json(self,condition) -> bool:
        if condition in self.readKey():
            return True
        else:
            return False
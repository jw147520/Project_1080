#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import socket
import struct
import pickle
import gzip

class Targets:
    def __init__(self, datasetPath, compressionLevel):
        self.pickleFile = None
        
        if datasetPath != None:
            self.pickleFile = gzip.open(datasetPath, mode='ab', compresslevel=compressionLevel)

    def parse(self, frame, jsonstr):
        try:
            dct = json.loads(jsonstr)
        except ValueError:
            return None
        
        dct['frame'] = frame
        if self.pickleFile != None:
            pickle.dump(dct, self.pickleFile)
        return dct

    # 새로운 save 함수.
    # Client.save_to_datafile 에서 호출한다.
    def save(self, frame, data):
        data['frame'] = frame  # data 는 parse 에서와 달리 이미 dictionary 이다.
        if self.pickleFile is not None:
            pickle.dump(data, self.pickleFile)

class Client:
    def __init__(self, ip='localhost', port=8000, datasetPath=None, compressionLevel=0):
        print('Trying to connect to DeepGTAV')
        
        self.targets = Targets(datasetPath, compressionLevel)

        try:
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.connect((ip, int(port)))
        except:
            print('ERROR: Failed to connect to DeepGTAV')
        else:
            print('Successfully connected to DeepGTAV')

    def sendMessage(self, message):
        jsonstr = message.to_json().encode('utf-8')
        try:
            self.s.sendall(len(jsonstr).to_bytes(4, byteorder='little'))
            self.s.sendall(jsonstr)
        except Exception as e:
            print('ERROR: Failed to send message. Reason:', e)
            return False
        return True

    def recvMessage(self):
        # _recvall() 에서는 frame 을 포함한 모든 정보를 읽어오는 듯 하다.
        # frame 을 안읽어오게 바꾸고 싶지만 자세한 구조를 모르기 때문에 패스.
        # 그냥 읽어온 frame 을 버리자.
        frame = self._recvall()
        if not frame: 
            print('ERROR: Failed to receive frame')     
            return None
        print(frame)
        data = self._recvall()
        if not data: 
            print('ERROR: Failed to receive message')       
            return None
        return self.targets.parse(frame, data.decode('utf-8'))

    # recvMessage() 에서는 message 를 받아오면서 target.parse 에 넘겨서 자동으로 dataset 에 저장한다.
    # frame 을 다른 방식으로 받아오기 때문에 바로 저장하면 안되므로 recvMessage_notSave() 를 작성하였다.
    def recvMessage_notSave(self):
        frame = self._recvall()
        if not frame:
            print('ERROR: Failed to receive frame')
            return None

        data = self._recvall()
        if not data:
            print('ERROR: Failed to receive message')
            return None

        try:
            dct = json.loads(data.decode('utf-8'))
        except ValueError:
            return None

        return dct

    # dataset 에 저장하는 함수를 새로 만들었다.
    def save_to_datafile(self, frame_img, data):
        self.targets.save(frame_img, data)

    def _recvall(self):
        #Receive first size of message in bytes
        data = b""
        while len(data) < 4:
            packet = self.s.recv(4 - len(data))
            if not packet: return None
            data += packet
        size = struct.unpack('I', data)[0]

        #We now proceed to receive the full message
        data = b""
        while len(data) < size:
            packet = self.s.recv(size - len(data))
            if not packet: return None
            data += packet
        return data

    def close(self):
        self.s.close()

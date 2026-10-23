import struct, pickle

'''
Pickle is good for localhost only. Its easy for MIM attack to tamper a pickle causing you to unpack an infected pickle.
'''

def recvall(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return bytes(buf)

'''
converting any data type into header+payload format
'''
def send_msg(sock, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL) # converting any python object into bytes for socket transfer. 
    sock.sendall(struct.pack("!I", len(data)) + data) # creates the header (informing of the size of the payload) into unsigned 32-bit int big-endian.
                                                        # Then sends the header + data. So [(data is n bytes long)+(the actual data)]

'''
unpacks the header+payload into original python object
'''
def recv_msg(sock):
    (n,) = struct.unpack("!I", recvall(sock, 4))    #reads the header to know the size of the payload (n) first 4 bytes of big-endian converted to int
    return pickle.loads(recvall(sock, n))   #reads the n bytes (payload) turning it back into the original python object
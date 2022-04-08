import socket

server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print("Server: Socket Created")

host = "localhost"
port = 5432

server_socket.bind((host,port))
print("Server: Socket connected to " + host)

mottat_data = " - Mottat"

while True:
    data, klient = server_socket.recvfrom(65535)
    
    if data:
        print("Server: Sender mottat data")
        server_socket.sendto(data + (mottat_data.encode()), klient)
import socket

def klient(x_koordinater,y_koordinater):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP

    host = 'localhost'     # IP
    port = 5433             # Port
    
    message = "X: " + str(x_koordinater) + "Y: " + str(y_koordinater)

    
    # print("Client: " + message)
    client_socket.sendto(message.encode(), (host,5432))
    
    data, server = client_socket.recvfrom(65535)
    data = data.decode()
    print("Client: " + data)
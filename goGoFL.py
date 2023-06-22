import requests

import server
#import client
import time

#This file should be used on the server
server.start_server()
time.sleep(5)
print("Server started")

URL_1 = "127.0.0.1:5000/startClient"
PARAMS_1 = {'url':"FL_Data/hh_01"}
x = client_1 = requests.get(url=URL_1, params=PARAMS_1)
print(f"client 1: {x.status_code}")

# URL_2 = "address2"
# PARAMS_2 = {'url':"FL_Data/hh_07"}
# client_2 = requests.get(url = URL_2, params = PARAMS_2)
#
# URL_3 = "address3"
# PARAMS_3 = {'url':"FL_Data/hh_01"}
# client_3 = requests.get(url = URL_3, params = PARAMS_3)

#
# client.start_client("FL_Data/hh_01")
# client.start_client("FL_Data/hh_07")
# client.start_client("FL_Data/hh_14")



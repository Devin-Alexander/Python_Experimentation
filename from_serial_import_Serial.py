# import Serial

# port = "COM13"
# baudrate = 115200

# with Serial(port=port, baudrate=baudrate, timeout=1) as port_serie:
#     if port_serie.isOpen():
#         port_serie.flush()
#         for i in range(20):
#             try:
#                 ligne = port_serie.readline()
#                 print(str(ligne))
#             except:
#                 print("Exception")
#         port_serie.close()



import serial
import numpy as np
from serial import Serial
ser = serial.Serial('COM13', 115200)


size_of_collection_array = 100

w_vals = np.zeros(size_of_collection_array)
i_vals = np.zeros(size_of_collection_array)
j_vals = np.zeros(size_of_collection_array)
k_vals = np.zeros(size_of_collection_array)

print("Hello World")


# ##############################
ser_bytes = ser.readline()
print(ser_bytes)
ser_bytes = ser.readline()
print(ser_bytes)



i = 0
num_digits_to_parse = 4

lock_acq_data = 0
lock_graph = 0

while i < size_of_collection_array:

    ser_bytes = ser.read() 

    if ser_bytes == b'$': # begin acquiring the data until one full set of quaternian data is collected
        lock_acq_data = 1 # lock to the data collection task only
        
        while lock_acq_data == 1:
            ser_bytes = ser.read() 

            if ser_bytes == b'^':
                w_vals[i] = float(ser.read(num_digits_to_parse)) # read 4 bytes and cast it into a float
                # print("w")
                # print(w_vals[i])

            if ser_bytes == b'~':
                i_vals[i] = float(ser.read(num_digits_to_parse)) # read 4 bytes and cast it into a float
                # print("i")
                # print(i_vals[i])

            if ser_bytes == b'!':
                j_vals[i] = float(ser.read(num_digits_to_parse)) # read 4 bytes and cast it into a float
                # print("j")
                # print(j_vals[i])

            if ser_bytes == b'@':
                k_vals[i] = float(ser.read(num_digits_to_parse)) # read 4 bytes and cast it into a float
                # print("k")
                # print(k_vals[i])
            
            if ser_bytes == b'#':
                i += 1
                lock_acq_data = 0 # set the lock to 0 to allow the while loop to break
                print("incremented value is:")
                print(i)
        

     

    # if ser_bytes == b'$':
    #     w_vals[i] = float(ser.read(num_digits_to_parse)) # read 4 bytes and cast it into a float
    #     i_vals[i] = float(ser.read(num_digits_to_parse)) # read 4 bytes and cast it into a float
    #     j_vals[i] = float(ser.read(num_digits_to_parse)) # read 4 bytes and cast it into a float
    #     k_vals[i] = float(ser.read(num_digits_to_parse)) # read 4 bytes and cast it into a float
    #     i += 1
    #     print("incremented value is:")
    #     print(i)




    


print("w_vals")
print(w_vals)
print("i_vals")
print(i_vals)
print("j_vals")
print(j_vals)
print("k_vals")
print(k_vals)



    # switcher(ser_bytes) {
    #     case '$': 
    #         i++
    #     break;

    #     case 


    
    
    # try:
    #     ser_bytes = ser.readline()
    #     print(ser_bytes)
    #     try:
    #         decoded_bytes = float(ser_bytes[0:len(ser_bytes)-2].decode("utf-8"))
    #         print(decoded_bytes)
    #     except:
    #         continue
    # except:
    #     print("Keyboard Interrupt")
    #     break











# import serial
# import time
# import csv
# import matplotlib 
# matplotlib.use("tkAgg")
# import matplotlib.pyplot as plt
# import numpy as np

# ser = serial.Serial('COM13', 115200)

# ser.flushInput()

# plot_window = 20
# y_var = np.array(np.zeros([plot_window]))

# plt.ion()
# fig, ax = plt.subplots()
# line, = ax.plot(y_var)
# while True:
#     try:
#         ser_bytes = ser.readline()
#         try:
#             decoded_bytes = float(ser_bytes[0:len(ser_bytes)-2].decode("utf-8"))
#             print(decoded_bytes)
#         except:
#             continue
#         with open("test_data.csv","a") as f:
#             writer = csv.writer(f,delimiter=",")
#             writer.writerow([time.time(),decoded_bytes])
#         y_var = np.append(y_var,decoded_bytes)
#         y_var = y_var[1:plot_window+1]
#         line.set_ydata(y_var)
#         ax.relim()
#         ax.autoscale_view()
#         fig.canvas.draw()
#         fig.canvas.flush_events()
#     except:
#         print("Keyboard Interrupt")
#         break

import serial
ser = serial.Serial('COM4', baudrate=9600 , timeout=1)
while 1:
    ser.write(4)
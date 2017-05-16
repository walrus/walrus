#!/usr/bin/python

# Simple script to connect to the Arduino and print whatever it sends over Serial

import serial, time, sys

# Open a Serial connection to the Arduino:
print "Connecting..."
try:
    arduino = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
except:
    print "Failed to connect on /dev/ttyACM0"
    sys.exit(2)

print "success"

msg = ""
while "Ready" not in msg:
    msg = arduino.readline()
    print msg

arduino.write('#')

while True:
    line = arduino.readline()
    print line

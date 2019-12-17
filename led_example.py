#!/usr/bin/python

import sys
import mraa
import time
import os


# Small delay to allow udev rules to execute (necessary only on up)
time.sleep(0.1)


# Loop
while True:

    #os.system("echo 1 > /sys/class/leds/upboard/green/brightness")

    with open("/sys/class/leds/upboard:green:/brightness", 'w') as green , open("/sys/class/leds/upboard:yellow:/brightness", 'w') as yellow , open("/sys/class/leds/upboard:red:/brightness","w") as red , open("/sys/class/leds/upboard:blue:/brightness","w") as blue:

       print("Turning leds off") 
       green.write("0")
       red.write("0")
       yellow.write("0")
       blue.write("0")

    time.sleep(0.5)

    with open("/sys/class/leds/upboard:green:/brightness", 'w') as green , open("/sys/class/leds/upboard:yellow:/brightness", 'w') as yellow , open("/sys/class/leds/upboard:red:/brightness","w") as red , open("/sys/class/leds/upboard:blue:/brightness","w") as blue:
       print("Turning leds on") 
       green.write("1")
       red.write("1")
       yellow.write("1")
       blue.write("1")

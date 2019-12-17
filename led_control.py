#!/usr/bin/python

import sys
import mraa
import time
import os



def led_control(led_status):
    """
    Given a string with len() == 3 and values 0-1

    *    The first char control the green led, 0 Off 1 On
    *    The second char control the green yellow, 0 Off 1 On
    *    The third char control the green red, 0 Off 1 On
    """
    with open("/sys/class/leds/upboard:green:/brightness", 'w') as green
    , open("/sys/class/leds/upboard:yellow:/brightness", 'w') as yellow
    , open("/sys/class/leds/upboard:red:/brightness","w") as red:
    
        assert len(led_status) == 3

        green.write(led_status[0])   
        yellow.write(led_status[1])
        red.write(led_status[2])



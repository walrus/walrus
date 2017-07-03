#!/usr/bin/python

# USAGE:
#
# Once running, press the following keys:
#
# '0..9' will note that the classification should have a 1 at the given index
# 'p' will note that the classification should be all zeroes
# 'x' will stop reading classifications and perform analysis


import serial, time, sys, curses

# Setup curses to catch keystrokes
window = curses.initscr()
window.nodelay(1)
window.keypad(1)

num_targets = 0
threshold = 0.5

classifications = []
targets = []

confusion = []

correct = 0
wrong = 0

def compute_classification(line):
    outputs = list(map(int(), line.split()))
    temp_threshold = threshold
    target = num_targets

    for j in range(len(outputs)):
        if outputs[j] > temp_threshold:
            target = j

    return target

def main(window):

    # Check argument(s).
    #
    # First (and only) arg is number of targets (max 9)

    global num_targets
    global threshold
    global classifications
    global targets
    global correct
    global wrong

    if len(sys.argv) < 3:
        print "Too few arguments; try again."
        sys.exit(1)
    else:
        num_targets = int(sys.argv[1])
        threshold = float(sys.argv[2])

    # Open a Serial connection to the Arduino:
    print "Connecting..."
    try:
        arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    except:
        print "Failed to connect on /dev/ttyACM0"
        sys.exit(2)

    print "success"

    # Read input from Serial and save the classifications/targets
    #  Will exit when x is pressed
    ch = -1

    while ch != ord('x'):
        line = arduino.readline()
        print line
        classification = compute_classification(line)
        print "Classification: {}".format(classification)
        classifications.append(classification)
        ch = window.getch()
        if ch != -1:
            target = 0
            if ch == ord('p'):
                target = num_targets
            else:
                target = ch - 30 # Reliant on ord being in range 30-39
            print "Target: {}".format(target)
            targets.append(target)

            if target == classification:
                correct +=1
            else:
                wrong += 1

        ch = -1


curses.wrapper(main)

# Now compute the various statistics

confusion = [[0 for x in range(num_targets)] for y in range(num_targets)]

for i in range(min(len(classifications), len(targets))):
    pass # do the thing

print "Correct classifications: {}".format(correct)
print "Incorrect classifications: {}".format(wrong)
print "Classification rate: {}".format(correct / correct + wrong)

sys.exit(0)
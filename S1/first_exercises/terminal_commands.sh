#!/bin/zsh

# used "chmod 700 terminal_commands.sh" to only change read/write/executable = 1 for owner 
x=10
echo The value of x is $x

for i in {1..10}
do
    python test.py
done


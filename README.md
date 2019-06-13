# Stegnography using python

Convert message into a binary sequence, and replace the least significant bits
(LSB) of the color pixels of a carrier image with the binary sequence
to hide the message.

Currently only work for PNG files with RGB channels.

The hiding and reading functions each has a numpy version to speed up
computation.

Optionally encrypt the message with AES encryption.

# Usage:

## Hide message into PNG image:

`python steg.py -c carrier_image.png -p message_file.txt`

New image is saved to `steg_carrier_image.png`.

To specify output image file name:

`python steg.py -c carrier_image.png -p message_file.txt -o output_file`

To add salt:

`python steg.py -c carrier_image.png -p message_file.txt -s 1`

To encrypt message with a password:

`python steg.py -c carrier_image.png -p message_file.txt -k my_password`


## Read message from image:

`python steg.py -c carrier_image.png`

Output message (if any) is saved to `steg_carrier_imag.txt`.

To specify output text file name:

`python steg.py -c carrier_image.png -o output_file`.

To decrypt message (if any) with a password:

`python steg.py -c carrier_image.png -k my_password`



# Example:

`python steg.py -c steg_bf.png`

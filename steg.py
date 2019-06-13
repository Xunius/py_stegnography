#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Stegnography using python

Convert message into a binary sequence, and replace the least significant bits
(LSB) of the color pixels of a carrier image with the binary sequence
to hide the message.

Currently only work for PNG files with RGB channels.

The hiding and reading functions each has a numpy version to speed up
computation.

Optionally encrypt the message with AES encryption.

Author: guangzhi XU (xugzhi1987@gmail.com; guangzhi.xu@outlook.com)
Update time: 2019-06-08 16:28:16.
'''


#--------Import modules-------------------------
import os
import sys
import argparse
from random import choice
from io import BytesIO
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
import base64
from PIL import Image
try:
    import numpy as np
    HAS_NUMPY=True
except:
    HAS_NUMPY=False



START_BUFFER = b'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE'
END_BUFFER   = b'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF'



class StegError(Exception):
    def __init__(self, text=None):
        if text is not None:
            self.message=text
    def __str__(self):
        return self.message


class FileNotFoundError(StegError):
    def __init__(self, text=None):
        if text is not None:
            self.message='File is not found.'
    def __str__(self):
        return self.message


class SizeOverFlowError(StegError):
    def __init__(self, text=None):
        if text is not None:
            self.message='Payload size is too big for carrier image.'
    def __str__(self):
        return self.message


class StartNotFoundError(StegError):
    def __init__(self, text=None):
        if text is not None:
            self.message='Start buffer not found.'
    def __str__(self):
        return self.message


class EndNotFoundError(StegError):
    def __init__(self, text=None):
        if text is not None:
            self.message='End buffer not found.'
    def __str__(self):
        return self.message



def getEncryptKey(password, key_len=16):
    '''Create an encryption key from password

    Uses the password itself as salt.
    '''

    key=PBKDF2(password, password, key_len)

    return key


def encrypt(message, key):
    '''Encrypt message using AES
    '''

    key_byte=bytearray(key)
    cipher=AES.new(key_byte, AES.MODE_CFB, key_byte)
    enc=cipher.encrypt(bytearray(message))
    enc=base64.urlsafe_b64encode(enc)

    return enc


def decrypt(enc, key):
    '''Decrypt message using AES
    '''

    key_byte=bytearray(key)
    decipher=AES.new(key_byte, AES.MODE_CFB, key_byte)
    dec=base64.urlsafe_b64decode(enc)
    dec=decipher.decrypt(dec)

    return dec


def byte2bin(barray):
    '''Convert a bytearray to binary string'''

    b_int=list(barray)
    b_bin=[bin(ii).lstrip('0b').zfill(8) for ii in b_int]
    b_bin=''.join(b_bin)

    return b_bin


def getImageFileSize(img, img_format='png'):
    '''Give an estimate of the image file without saving
    '''

    img_f=BytesIO()
    img.save(img_f, img_format)
    fsize=img_f.tell()/1000.  # kb

    return fsize



def changeLSB(old_byte, new_bit, bit=-1):
    '''Change the least significant bit of a byte
    '''

    b = list(bin(old_byte))
    b[bit] = new_bit

    return int(''.join(b),2)


def getSalt(n):
    '''Create random binary salt generator

    Args:
        n (int): length of random salt.
    Returns:
        salt (generator): a generator to create a random binary sequence
                          with length <n>.
    '''

    salt=(choice('01') for i in xrange(n))
    return salt



def clearLastNBits(array, n):
    '''Clear the last n bits of each number in array

    Args:
        array (ndarray): input int array to change.
        n (int): number of last bits to clear.
    Returns:
        result (ndarray): array with its last n bits in each number set to 0.
    '''

    # shape of array is, e.g. (n, m, 3)

    binarray=np.reshape(array, array.shape+(1,)) # (n, m, 3, 1)
    binarray=binarray & (1 << np.arange(8)[::-1]) # (n, m, 3, 8)
    binarray=(binarray > 0).astype('int')  # (n, m, 3, 8), binary
    # clear last n bits
    binarray[..., -n:]=0
    # binary to int
    result=np.packbits(binarray, axis=-1)[...,-1]  # (n, m, 3)

    return result


def getLastNBits(array, n):
    '''Return the last n bits of each number in array

    Args:
        array (ndarray): input int array to change.
        n (int): number of last bits to clear.
    Returns:
        result (ndarray): array with its last n bits in each number returned.
    '''

    binarray=np.reshape(array, array.shape+(1,)) # (n, m, 3, 1)
    binarray=binarray & (1 << np.arange(8)[::-1]) # (n, m, 3, 8)
    binarray=(binarray > 0).astype('int')  # (n, m, 3, 8), binary
    # get last n bits
    result=binarray[..., -n:]

    return result





class Steg(object):

    def __init__(self, carrier_path, output, max_num_bits=3, salt=False,
            encrypt_key=None):

        self.carrier_path=carrier_path
        self.output=output
        self.max_num_bits=max_num_bits
        self.salt=salt
        self.encrypt_key=encrypt_key

        self.analyzeImage()


    def analyzeImage(self):

        if not os.path.exists(self.carrier_path):
            raise FileNotFoundError("Carrier file is not found: %s"\
                    %self.carrier_path)

        filename, self.image_type=os.path.splitext(self.carrier_path)
        try:
            self.carrier=Image.open(self.carrier_path)
        except:
            raise Exception("Failed to open carrier image.")
        else:
            self.image_size=self.carrier.size[1] * self.carrier.size[0]
            # Gets the image mode, hopefully this is L, RGB, or RGBA
            self.image_mode=self.carrier.mode
            if self.image_mode not in ['RGB',]:
                raise Exception("Image mode not supported.")

            print('# <analyzeImage>: carrier size = ', self.carrier.size)
            print('# <analyzeImage>: carrier mode = ', self.image_mode)


    def bitsRequired(self, mode):
        '''Estimate the required number of LSBs and channels to hold payload

        NOTE: need to call this after calling readPayload(), as the encrypted
        text may have different size.
        '''

        if mode=='RGB':
            n_channels=3
        elif mode=='RGBA':
            n_channels=3
        elif mode=='L':
            n_channels=1

        # unit: bytes
        buffer_size=len(START_BUFFER) + len(END_BUFFER)
        payload_size=os.path.getsize(self.payload_path) + buffer_size
        slab_size=self.image_size

        # number of least sig bits required
        n_bits=int(float(payload_size)/slab_size/n_channels)+1
        if n_bits>self.max_num_bits:
            raise SizeOverFlowError("Carrier image is not big enough to hold payload.")

        # if requiring > 1 bits, already using all channels
        if n_bits>1:
            n_chn=n_channels
        else:
            n_chn=1

        return n_bits, n_chn


    def readPayload(self, path):
        '''Read in payload texts
        '''

        self.payload_path=path

        if not os.path.exists(path):
            raise FileNotFoundError('<path> not found: %s' %path)

        try:
            with open(path, 'rb') as fin:
                self.payload_text=fin.read()
        except:
            raise Exception("Failed to read payload.")
        else:
            # encrypt message if encrypt_key is given
            if self.encrypt_key is not None:
                enc_key=getEncryptKey(self.encrypt_key)
                self.payload_text=encrypt(self.payload_text, enc_key)

            # add buffers, convert to binary
            self.payload_text='%s%s%s' %(START_BUFFER, self.payload_text,
                    END_BUFFER)
            self.payload_text_bytes=bytearray(self.payload_text)
            self.payload_text_bin=byte2bin(self.payload_text_bytes)


    def getPixel(self, n_bits, n_channels):
        '''Iterator to loop through all pixels in the carrier image

        Args:
            n_bits (int): number of LSBs to loop through.
            n_channels (int): number of color channels to loop though.
        Returns:
            [fgr, fgg, fgb]: R, G, B colors in a list.
            row (int): row index of yielded pixel.
            col (int): col index of yielded pixel.
            -cjj (int): color channel index of yielded pixel.
            -bjj (int): LSB bit index of yielded pixel.

        The scanning order is:
            columns -> rows -> channels (in order of BGR) -> LSB bis (in order
            of last, 2nd last, 3rd last bit).

        As it might scan the image more than once, create a cache dict
        to hold previous results.
        '''

        cache={}

        nrow=self.carrier.size[1]
        ncol=self.carrier.size[0]

        for bii in range(1, n_bits+1):
            for cjj in range(1, n_channels+1):
                for row in xrange(nrow):
                    for col in xrange(ncol):
                        if (row, col) in cache:
                            yield cache[(row, col)], row, col, -cjj, -bii
                        else:
                            fgr,fgg,fgb = self.carrier.getpixel((col,row))
                            cache[(row, col)]=[fgr, fgg, fgb]
                            yield [fgr, fgg, fgb], row, col, -cjj, -bii



    def getPixelLSB(self, n_bits, n_channels):
        '''Iterator to loop though usable LSBs in the carrier image

        Args:
            n_bits (int): number of LSBs to loop through.
            n_channels (int): number of color channels to loop though.
        Returns:
            result (str): single binary digit of the next LSB to be changed
                          in the carrier image.

        The scanning order is:
            columns -> rows -> channels (in order of BGR) -> LSB bis (in order
            of last, 2nd last, 3rd last bit).
        '''

        cache={}

        nrow=self.carrier.size[1]
        ncol=self.carrier.size[0]

        for bii in range(1, n_bits+1):
            for cjj in range(1, n_channels+1):
                for row in xrange(nrow):
                    for col in xrange(ncol):
                        if (row, col) in cache:
                            yield cache[(row, col)][-cjj][-bii]
                        else:
                            fgr,fgg,fgb = self.carrier.getpixel((col,row))
                            r_bin=bin(fgr)
                            g_bin=bin(fgg)
                            b_bin=bin(fgb)

                            cache[(row, col)]=(r_bin, g_bin, b_bin)
                            result=[r_bin, g_bin, b_bin][-cjj][-bii]
                            yield result



    def hideInfo_RGB(self, salt=None):
        '''Hide information into RGB carrier image

        Args:
            salt (bool): whether to add random salt or not. If None, will use
                         the attribute of the Steg obj.
        '''

        if salt is None:
            salt=self.salt

        n_bits, n_channels=self.bitsRequired('RGB')
        print('# <hideInfo_RGB>: n_bits=', n_bits, 'n_channels=', n_channels)

        # make a copy of the carrier
        new_img = self.carrier.copy()

        # get payload stream
        bitstream=iter(self.payload_text_bin)

        # get pixel stream
        pixel_gen=self.getPixel(n_bits, n_channels)

        cache={}  # save visited pixels
        while True:
            try:
                bii=next(bitstream)
            except StopIteration:
                break
            else:
                pii, row, col, channel, bit=pixel_gen.next()
                if (row, col) in cache:
                    # pii is [r, g, b] bytes at (row, col)
                    pii=cache[(row, col)]
                # change the bit at <bit> in channel <channel>
                pii[channel]=changeLSB(pii[channel], bii, bit)
                cache[(row, col)]=pii

        # assign new pixels
        # have to do this after all bits altered, as a pixel may be altered
        # more than once
        for (row, col), pii in cache.items():
            new_img.putpixel((col, row), tuple(pii))

        if salt:
            # compute salt size
            salt_size=self.image_size*n_channels*n_bits -\
                    len(self.payload_text_bin)
            saltstream=getSalt(salt_size)

            counter=0
            carrier_file_size=os.path.getsize(self.carrier_path)/1000  # kb
            while True:
                try:
                    bii=next(saltstream)
                except StopIteration:
                    break
                else:
                    pii, row, col, channel, bit=pixel_gen.next()
                    if (row, col) in cache:
                        pii=cache[(row, col)]
                    pii[channel]=changeLSB(pii[channel], bii, bit)
                    cache[(row, col)]=pii

                    new_img.putpixel((col, row), tuple(pii))
                    counter+=1
                    # check every k
                    if counter%1000==1:
                        fsize=getImageFileSize(new_img, 'png')
                        # if file size differ too much, quit salting
                        if abs(fsize-carrier_file_size)>=5:
                            print('# <hideInfo_RGB>: current files size =', fsize)
                            break

        # save output
        output_file_type = self.image_type
        output_file_path='%s%s' %(self.output, output_file_type)
        new_img.save(output_file_path, output_file_type.replace('.',''))
        print('# <hideInfo_RGB>: New image created:', output_file_path)

        '''
        go=True
        for fgr, fgg, fgb, row, col in self.getPixel():
            if not go:
                break
            pixel_colors=[fgr, fgg, fgb]
            print('# <hideInfo_RGB>: pixel at (%d, %d) = %s' %(row, col,
                str(pixel_colors)))
            for ii in range(3):
                try:
                    bii=next(bitstream)
                except StopIteration:
                    go=False
                else:
                    pixel_colors[ii]=changeLSB(pixel_colors[ii], bii)

            new_img.putpixel((col, row), tuple(pixel_colors))
        '''

        '''
        for row in xrange(self.carrier.size[0]):
            for col in xrange(self.carrier.size[1]):
                # get the value for each byte of each pixel in the original image
                fgr,fgg,fgb = self.carrier.getpixel((col,row))

                # get the new lsb value from the bitstream
                rb = next(bitstream)
                # modify the original byte with our new lsb
                fgr = changeLSB(fgr, rb)

                gb = next(bitstream)
                fgg = changeLSB(fgg, gb)

                bb = next(bitstream)
                fgb = changeLSB(fgb, bb)
                # add pixel with modified values to new image
                new_img.putpixel((col, row), (fgr, fgg, fgb))
        '''
        return



    def hideInfo_RGB_numpy(self):
        '''Hide information into RGB carrier image, numpy version

        '''

        n_bits, n_channels=self.bitsRequired('RGB')
        print('# <hideInfo_RGB_numpy>: n_bits=', n_bits, 'n_channels=', n_channels)

        carrier_arr=np.asarray(self.carrier)

        # clear last n bits
        carrier_trunc=clearLastNBits(carrier_arr, n_bits)

        nrow=self.carrier.size[1]
        ncol=self.carrier.size[0]
        slab_size=nrow*ncol

        # get payload stream
        bitstream=np.array(list(self.payload_text_bin)).astype('uint8')

        for bii in range(1, n_bits+1):
            for cjj in range(1, n_channels+1):
                #slabij=np.random.randint(0, 2, size=slab_size)
                slabij=np.zeros(slab_size)

                bitsij=bitstream[((bii-1)*n_channels + (cjj-1)*n_channels)*slab_size:\
                        ((bii-1)*n_channels + cjj*n_channels)*slab_size]
                slabij[:len(bitsij)] = bitsij
                slabij=slabij.reshape((nrow, ncol))

                # add new bits to carrier
                carrier_trunc[:,:,-cjj]+=(slabij*2**(bii-1)).astype('uint8')

        # save image
        new_img=Image.fromarray(carrier_trunc)
        output_file_type = self.image_type
        output_file_path='%s%s' %(self.output, output_file_type)
        new_img.save(output_file_path, output_file_type.replace('.',''))
        print('# <hideInfo_RGB>: New image created:', output_file_path)

        return



    def readInfo_RGB(self):
        '''Read message from carrier, for RGB image
        '''

        bins=''
        s_buffer_size=len(START_BUFFER)*8   # bits
        e_buffer_size=len(END_BUFFER)*8
        # check buffer every number of bits
        buffer_check=max(s_buffer_size, e_buffer_size)
        # convert buffers to binary
        s_buffer_bin=byte2bin(bytearray(START_BUFFER))
        e_buffer_bin=byte2bin(bytearray(END_BUFFER))

        n=0
        n_channels=3
        start_idx=-1   # index of START_BUFFER
        end_idx=-1     # index of END_BUFFER

        # loop through LSBs
        for pii in self.getPixelLSB(self.max_num_bits, n_channels):
            n+=1
            bins+=pii

            if n%buffer_check==0:
                if start_idx==-1:
                    s_idx=bins.find(s_buffer_bin)
                    if s_idx!=-1:
                        start_idx=s_idx

                if end_idx==-1:
                    e_idx=bins.find(e_buffer_bin)
                    if e_idx!=-1:
                        end_idx=e_idx

            if end_idx!=-1:
                break

        # do one more check, as buffer_check may not evenly divide stream size
        if start_idx==-1:
            s_idx=bins.find(s_buffer_bin)
            if s_idx==-1:
                raise StartNotFoundError()

        if end_idx==-1:
            e_idx=bins.find(e_buffer_bin)
            if e_idx==-1:
                raise EndNotFoundError()

        bins=bins[start_idx+s_buffer_size : end_idx]
        # group into 8 bits
        str_bytes=[bins[i:i+8] for i in range(0, len(bins), 8)]

        # binary to str
        hidden=''
        for ii in xrange(len(str_bytes)):
            cii=chr(int(str_bytes[ii],2))
            hidden+=cii

        # decrypt
        if self.encrypt_key is not None:
            enc_key=getEncryptKey(self.encrypt_key)
            hidden=decrypt(hidden, enc_key)

        output_file_path='%s.txt' %self.output
        with open(output_file_path, 'w') as fout:
            fout.write(hidden)

        print('# <readInfo_RGB>: Message wrote to:', output_file_path)

        return hidden



    def readInfo_RGB_numpy(self):
        '''Read message from carrier, for RGB image, numpy version
        '''

        carrier_arr=np.asarray(self.carrier)

        # get LSBs
        lastbits=getLastNBits(carrier_arr, self.max_num_bits)
        # get correct order: (bits, channels, row, col)
        lastbits=np.transpose(lastbits, (3,2,0,1))
        # reverse bits order
        lastbits=lastbits[::-1, :,:,:]
        # reverse channel order
        lastbits=lastbits[:, ::-1, :,:]
        lastbits=lastbits.flatten()
        # pad to multiples of 8
        n=int(len(lastbits)/8)+1
        lastbits=np.pad(lastbits, (0, n*8-len(lastbits)), 'constant',
                constant_values=0)
        lastbits=lastbits.reshape((n, 8))

        # binary to int
        int_stream=np.packbits(lastbits, axis=-1).squeeze()

        hidden=''
        s_buffer_size=len(START_BUFFER)
        e_buffer_size=len(END_BUFFER)
        buffer_check=max(s_buffer_size, e_buffer_size)

        n=0
        start_idx=-1   # index of START_BUFFER
        end_idx=-1     # index of END_BUFFER

        for ii in xrange(len(int_stream)):
            cii=int_stream[ii]
            cii=chr(cii)
            hidden+=cii

            n+=1
            if n%buffer_check==0:
                if start_idx==-1:
                    s_idx=hidden.find(START_BUFFER)
                    if s_idx!=-1:
                        start_idx=s_idx

                if end_idx==-1:
                    e_idx=hidden.find(END_BUFFER)
                    if e_idx!=-1:
                        end_idx=e_idx

            if end_idx!=-1:
                break

        # do one more check, as buffer_check may not evenly divide stream size
        if start_idx==-1:
            s_idx=hidden.find(START_BUFFER)
            if s_idx==-1:
                raise StartNotFoundError()

        if end_idx==-1:
            e_idx=hidden.find(END_BUFFER)
            if e_idx==-1:
                raise EndNotFoundError()

        hidden=hidden[start_idx+s_buffer_size : end_idx]

        # decrypt
        if self.encrypt_key is not None:
            enc_key=getEncryptKey(self.encrypt_key)
            hidden=decrypt(hidden, enc_key)

        output_file_path='%s.txt' %self.output
        with open(output_file_path, 'w') as fout:
            fout.write(hidden)

        print('# <readInfo_RGB_numpy>: Message wrote to:', output_file_path)

        return hidden






def main(args):

    carrier=args.carrier
    payload=args.payload
    password=args.key
    salt=args.salt
    output=args.output

    if output is None:
        output='steg_%s' %os.path.splitext(carrier)[0]

    print('# <main>: carrier = ', carrier)
    print('# <main>: payload = ', payload)
    print('# <main>: password = ', password)
    print('# <main>: salt = ', salt)
    print('# <main>: output = ', output)

    stg=Steg(carrier, salt=salt, encrypt_key=password, output=output)

    if payload is not None:
        stg.readPayload(payload)
        if salt:
            stg.hideInfo_RGB(salt=True)
        else:
            if HAS_NUMPY:
                stg.hideInfo_RGB_numpy()
            else:
                stg.hideInfo_RGB(salt=False)

    else:
        if HAS_NUMPY:
            hidden=stg.readInfo_RGB_numpy()
        else:
            hidden=stg.readInfo_RGB()

        print('# <main>: hidden message:')
        print(hidden)

    return


#-------------Main---------------------------------
if __name__=='__main__':


    parser=argparse.ArgumentParser()
    parser.add_argument('-c', dest='carrier', type=str, default=None,
            help='Path to the carrier file.')
    parser.add_argument('-p', dest='payload', type=str, default=None,
            help='Path to the payload file.')
    parser.add_argument('-k', dest='key', type=str, default=None,
            help='Encryption password.')
    parser.add_argument('-s', dest='salt', type=bool, default=False,
            help='Add salt or not.')
    parser.add_argument('-o', dest='output', type=str, default=None,
            help='''Output file path. If payload is given, this is the ouput
            image file name (without extension). If payload is not given,
            this is the text file if message is found.''')

    args = parser.parse_args()

    if args.carrier is None:
        print('[!] No carrier supplied.')
        parser.print_help()
        sys.exit(1)

    rec=main(args)


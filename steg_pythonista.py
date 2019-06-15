'''Stegnography using python

Script for Pythonista

Author: guangzhi XU (xugzhi1987@gmail.com; guangzhi.xu@outlook.com)
Update time: 2019-06-08 16:28:16.
'''

import ui
import os
import clipboard
from console import hud_alert
import dialogs
import photos
import io
from steg import StegCore, StartNotFoundError



def run(sender):
    stegobj.run(sender)

def copyText(sender):
    stegobj.copyText(sender)

def pasteText(sender):
    stegobj.pasteText(sender)

def clearText(sender):
    stegobj.clearText(sender)

@ui.in_background
def pickImage(sender):
    stegobj.pickImage(sender)

def img2Bytes(img_in):
    with io.BytesIO() as bio:
        img_in.save(bio, 'PNG')
        img_out=bio.getvalue()
    return img_out



class StegView(object):

    def __init__(self, view):
        self.view=view
        self.subviews=view.subviews
        self.tv=self.findChild('scrollview1').subviews[0]
        self.iv=self.findChild('imageview1')
        self.run_btn=self.findChild('run_btn')
        self.img_pick_btn=self.findChild('img_pick_btn')
        self.paste_text_tn=self.findChild('paste_text_btn')
        self.copy_text_tn=self.findChild('copy_text_btn')
        self.clear_text_tn=self.findChild('clear_text_btn')
        self.password_tf=self.findChild('password_tf')

        self.carrier=None

    def findChild(self, name):
        for sii in self.subviews:
            if sii.name==name:
                return sii

    def copyText(self, sender):
        '@type sender: ui.Button'
        #tv = sender.superview['scrollview1'].subviews[0]
        tv=self.tv
        text = tv.text
        clipboard.set(text)
        hud_alert('Copied')

    def pasteText(self, sender):
        '@type sender: ui.Button'
        #tv = sender.superview['scrollview1'].subviews[0]
        tv=self.tv
        text = tv.text
        new=clipboard.get()
        new=text+' '+new
        st=tv.selected_range
        tv.replace_range(st, new)

    def clearText(self, sender):
        '@type sender: ui.Button'
        #tv = sender.superview['scrollview1'].subviews[0]
        tv=self.tv
        tv.text=''

    def pickImage(self, sender):
        '@type sender: ui.Button'
        i = dialogs.alert('Image', '', 'Select from Photos')
        iv=self.iv
        #iv=sender.superview['imageview1']
        if i == 1:
            img = photos.pick_image()
            self.carrier=img
            uiimg = img2Bytes(img)

            uiimg=ui.Image.from_data(uiimg)
            iv.image=uiimg

    def run(self, sender):
        if self.carrier is None:
            hud_alert('Select a carrier image first.')
            return
        #pwd=self.password_tf.text
        #pwd=pwd.encode('ascii')
        #if len(pwd)==0:
        	#pwd=None
        pwd=None

        stegcore=StegCore(self.carrier, self.tv.text,
                salt=False, encrypt_key=pwd)

        if len(self.tv.text)==0:
            try:
                hidden=stegcore.readInfo_RGB_numpy()
            except StartNotFoundError:
                hud_alert('No message found.')
            else:
                self.tv.text=hidden
        else:
            stegcore.processPayload(self.tv.text)
            new_img=stegcore.hideInfo_RGB_numpy()
            img_path='nee_steg.png'
            new_img.save(img_path)
            photos.create_image_asset(img_path)
            os.remove(img_path)
            hud_alert('New image created.')


def main():

    v = ui.load_view('steg_ui')
    global stegobj
    stegobj=StegView(v)
    v.present('sheet')

if __name__=='__main__':
    main()

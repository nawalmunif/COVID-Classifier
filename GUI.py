from tkinter import * 
import os
from PIL import ImageTk
import PIL.Image 
from tkinter import filedialog 
from predict_model import *
from process_image_for_gui import *
window = Tk(className='Covid Prediction Test')
window.geometry("700x700")
frame0 = Frame(window)
Label(frame0, 
		 text="COVID Classifier",
		 fg = "black",
		 font = "Times 30 bold",
        ).grid()

# canvas = Canvas(window, width = 500, height = 500)      
# canvas.pack()   
# path = r"H:\COVID\COVID-Classifier\test.jpg"  
# img = ImageTk.PhotoImage(file=path)  
# canvas.create_image(200,200, anchor='c', image=img) 


# def imageTrue():
#     img = ImageTk.PhotoImage(file=path)  
#     #canvas.create_image(200,200, anchor='c', image=img)
#     #print(path)
#     imgLabel = Label(window, image=img)
#     imgLabel.pack(side=TOP)
# def printName():
#     path = entry.get()
#     #imageTrue(path)


def label_scores(Y_Score):
    global label1_score,label2_score,label3_score
    label1_score["text"] = str(round((Y_Score[0][0]*100),3))+'%'
    label2_score["text"] = str(round((Y_Score[0][1]*100),3))+'%'
    label3_score["text"] = str(round((Y_Score[0][2]*100),3))+'%'

Y_Score = []
# def destroy_labels():
#     label1_score.destroy()
#     label2_score.destroy()
#     label3_score.destroy()


def browseFiles():
    global Y_Score
    Y_Score = [] 
    filename = filedialog.askopenfilename(initialdir = "/", 
                                          title = "Select a File", 
                                          filetypes = (('all files', '.*'),
                                          ('JPG files', '*.jpg'),
                                          ('PNG files', '*.png'))) 
    pre_image = processing_image(filename)
    Y_Score = extractFeaturesTest(pre_image)
    #Y_Score = [0.99,0.008,0.097]
    label_scores(Y_Score) 
    img = PIL.Image.open(filename)
    img = img.resize((256, 256), PIL.Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(frame4, image = img) 
    panel.image = img
    panel.grid(column=2, row = 3)
    

frame1 = Frame(window)
frame2 = Frame(window)
frame3 = Frame(window)
frame4 = Frame(window)
label1= Label(frame1,text="Covid",justify=LEFT,padx=50)
label1.grid()
label2= Label(frame1,text="Normal",justify=LEFT,padx=100)
label2.grid()

label3= Label(frame1,text="Pneumonia",justify=LEFT,padx=100)
label3.grid()


label1_score= Label(frame2,text='',justify=LEFT,padx=50)
label1_score.grid()

label2_score= Label(frame2,text='',justify=LEFT,padx=100)
label2_score.grid()

label3_score= Label(frame2,text='',justify=LEFT,padx=100)
label3_score.grid()

button3 = Button(frame3, text = "Upload Image" ,bg='white', fg ='black',command = browseFiles)
button3.grid()

frame0.grid(column = 0, row = 0)
frame1.grid(column = 0, row = 1)
frame2.grid(column = 2, row = 1)
frame3.grid(column = 1, row = 2)
frame4.grid(column = 0, row = 3)

window.mainloop()
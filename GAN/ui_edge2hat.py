#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import matplotlib.pyplot as plt
# import numpy as np
# tmp = np.zeros((100, 100))
# plt.imshow(tmp, cmap='gray')
# plt.show()


# In[2]:


import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
from tkinter import Scale, HORIZONTAL, Label
from tkinter.filedialog import askopenfilename, asksaveasfilename
import scipy.ndimage
import os
import numpy as np
import sys


# In[3]:


import torch
import torch.nn as nn
import torchvision.transforms as transforms


# In[4]:


from model_edge2hats import Generator, ResNetBlock


# In[5]:


# define global variables
global resolution, draw_border
global image, draw_im, prev_image
global eraser # eraser icon
global last_x, last_y
global thickness # pen thickness
global mode
global template_photo
global creativity
global transform_edge, nz, ngf, model, mean
global draw_button, erase_button
global image_array


# In[6]:


def main():
    global resolution, draw_border, image, draw_im, eraser, last_x, last_y, thickness, mode, template_photo
    global draw_button, erase_button
    root = tk.Tk()
    # main window
    root.title("Edge2Hats")
    root.geometry("%sx%s"%(resolution + 100, resolution + 100))
    root.grid_rowconfigure(4, weight = 1)
    root.grid_columnconfigure([i for i in range(5)], weight = 1)
    
    # canvas
    frame = tk.Frame(root)
    frame.grid(row = 4, column = 0, columnspan = 5, sticky="NSEW")
    canvas = tk.Canvas(frame, bg = "white", width = resolution, height = resolution)
    canvas.bind("<B1-Motion>", lambda event, canvas = canvas:draw(canvas, event))
    canvas.bind("<ButtonRelease-1>", lambda event, canvas = canvas:reset_position(canvas))
    canvas.pack()
    frame.bind("<Configure>", lambda event, canvas = canvas:resize(event, canvas))

    # image cache
    image = Image.new('L', (resolution, resolution), 255)
    draw_im = ImageDraw.Draw(image)
    mode = "draw"

    # pen thickness
    thickness_scale = Scale(root, from_ = 1, to = 20, 
                            orient = HORIZONTAL, command = update_thickness)
    thickness_scale.set(thickness)
    thickness_label = Label(root, text = "thickness")
    thickness_label.grid(row = 3, column = 0)
    thickness_scale.grid(row = 3, column = 1)

    # eraser
    erase_button = tk.Button(root, text = "eraser", command = lambda:switch_mode("erase"))
    erase_button.grid(row = 2, column = 1)
    
    # pen
    draw_button = tk.Button(root, text = "pen", command = lambda:switch_mode("draw"))
    draw_button.grid(row = 2, column = 0)
    draw_button.config(state = "disabled")
    
    # template
    # template_label = Label(root, text = "Select template")
    template_button = tk.Button(root, text = "load", command = lambda:target_template(canvas))
    # template_label.grid(row = 0, column = 0)
    template_button.grid(row = 0, column = 0)

    # save image
    save_button = tk.Button(root, text = "save", command = lambda:save_image(image))
    save_button.grid(row = 1, column = 0)

    # creativity
    creativity_scale = Scale(root, from_ = 0, to = 20,
                            orient = HORIZONTAL, command = update_creativity)
    creativity_scale.set(creativity)
    creativity_label = Label(root, text = "creativity")
    creativity_label.grid(row = 3, column = 2)
    creativity_scale.grid(row = 3, column = 3)    

    # color balance
    color_instruction_label = Label(root, text = "color scale")
    color_instruction_label.grid(row = 0, column = 3)

    red_label = Label(root, text = "cyan - red")
    red_label.grid(row = 2, column = 2)
    red_scale = Scale(root, from_ = -color_balance_range, to = color_balance_range, orient = HORIZONTAL)
    red_scale.set(0)
    red_scale.bind("<ButtonRelease-1>", lambda event:update_color_balance(red_scale.get(),
                                                                         "red",
                                                                         canvas))
    red_scale.grid(row = 1, column = 2)
    red_scale.config(state = "disabled")

    green_label = Label(root, text = "magenta - green")
    green_label.grid(row = 2, column = 3)
    green_scale = Scale(root, from_ = -color_balance_range, to = color_balance_range, orient = HORIZONTAL)
    green_scale.set(0)
    green_scale.bind("<ButtonRelease-1>", lambda event:update_color_balance(green_scale.get(),
                                                                         "green",
                                                                         canvas))
    green_scale.grid(row = 1, column = 3)
    green_scale.config(state = "disabled")

    blue_label = Label(root, text = "yellow - blue")
    blue_label.grid(row = 2, column = 4)
    blue_scale = Scale(root, from_ = -color_balance_range, to = color_balance_range, orient = HORIZONTAL)
    blue_scale.set(0)
    blue_scale.bind("<ButtonRelease-1>", lambda event:update_color_balance(blue_scale.get(),
                                                                         "blue",
                                                                         canvas))
    blue_scale.grid(row = 1, column = 4)
    blue_scale.config(state = "disabled")

    color_balance_list = [red_scale, green_scale, blue_scale]
    # generate command
    gen_button = tk.Button(root, text = "edge2hats", 
                           command = lambda:predict(canvas, 
                                                    enable_button_lists + [erase_button, draw_button], 
                                                    gen_button,
                                                    color_balance_list))
    gen_button.grid(row = 3, column = 4)

    # cancel command
    cancel = tk.Button(root, text = "cancel", 
                       command = lambda:cancel_command(canvas, 
                                                       enable_button_lists + [gen_button],
                                                       color_balance_list))
    cancel.grid(row = 0, column = 1)

    # clear
    enable_button_lists = [template_button]
    clear_button = tk.Button(root, text = "clear", 
                             command = lambda:clear_canvas(canvas, enable_button_lists + [gen_button], color_balance_list))
    clear_button.grid(row = 1, column = 1)

    # mainloop
    root.mainloop()


# In[7]:


def channel_update(array, adjustment, channel_idx):
    array[:, :, channel_idx] = image_array[:, :, channel_idx]
    if adjustment > 0:
        array[:, :, channel_idx] = array[:, :, channel_idx] + (255 - array[:, :, channel_idx]) * adjustment / 100
    elif adjustment < 0:
        array[:, :, channel_idx] = array[:, :, channel_idx] + array[:, :, channel_idx] * adjustment / 100
    return array

def update_color_balance(value, channel, canvas):
    if image_array is None:
        return
    global template_photo, image
    image_array2 = np.array(image)
    if channel == "red":
        image_array2 = channel_update(image_array2.astype(np.float32), (value), 0)
    elif channel == "green":
        image_array2 = channel_update(image_array2.astype(np.float32), (value), 1)
    else:
        image_array2 = channel_update(image_array2.astype(np.float32), (value), 2)
    image_array2 = np.clip(image_array2, 0, 255).astype(np.uint8)
    image = Image.fromarray(image_array2)
    template_photo = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, anchor = "nw", image = template_photo)


# In[8]:


def cancel_command(canvas, enable_button_list, disable_button_list):
    for button in enable_button_list:
        button.config(state = "normal")
    for button in disable_button_list:
        button.config(state = "disabled")
    switch_mode("draw")
    canvas.bind("<B1-Motion>", lambda event, canvas = canvas:draw(canvas, event))
    global image, template_photo, image_array, prev_image, draw_im
    if prev_image:
        image = prev_image.copy()
        draw_im = ImageDraw.Draw(image)
        template_photo = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor = "nw", image = template_photo)
        prev_image = None
        image_array = None


# In[9]:


def update_creativity(val):
    global creativity
    creativity = int(val)


# In[10]:


def tensor2image(tensor):
    low = float(tensor.min())
    high = float(tensor.max())
    tensor = tensor.clamp_(min = low, max = high)
    tensor = tensor.sub_(low).div_(max(high - low, 1e-5))
    image = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    zoom_factor = [resolution / image.shape[0], resolution / image.shape[0], 1]
    image = scipy.ndimage.zoom(image, zoom_factor)
    return image
    
def plot_prediction(array, canvas):
    global template_photo, image_array, image
    image_array = (array.clip(0, 255)).astype(np.uint8)
    image = Image.fromarray(image_array)
    template_photo = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, anchor = "nw", image = template_photo)

def predict(canvas, disable_button_lists, gen_button, enable_button_lists):
    gen_button.config(state = "disabled")
    for button in disable_button_lists:
        button.config(state = "disabled")
    for button in enable_button_lists:
        button.config(state = "normal")
        button.set(0)
    canvas.unbind("<B1-Motion>")
    global prev_image
    prev_image = image.copy()
    image_array = np.array(image)
    image_tensor = transform_edge(Image.fromarray(image_array))
    image_tensor = image_tensor.unsqueeze(0)
    # fixed_noise = torch.randn(1, nz, 1, 1) * (creativity + 0.0001) + mean
    fixed_noise = torch.randn(1, nz, 1, 1) + creativity
    model.eval()
    with torch.no_grad():
        generated = model(fixed_noise, image_tensor)
    image_array = tensor2image(generated[0])
    plot_prediction(image_array, canvas)
    


# In[11]:


# save function
def save_image(image):
    filename = asksaveasfilename(defaultextension = ".png", filetypes = [("PNG files", "*.png")])
    try:
        image.save(filename)
    except Exception as e:
        tk.messagebox.showerror("Error", "Failed to save image. Error is %s"%e)


# In[12]:


# restrict the canvas to a fixed size
def resize(event, canvas):
    global template_photo
    template_photo = ImageTk.PhotoImage(image)
    canvas.config(width = resolution, height = resolution)
    canvas.create_image(0, 0, anchor = "nw", image = template_photo)


# In[13]:


# draw and erase functions
# given start and end points, create lines and store them in the cache
def update_line(x1, y1, x2, y2, canvas):
    global mode, draw_im
    if mode == "draw":
        canvas.create_line(x1, y1, x2, y2, fill='black')
        draw_im.line([x1, y1, x2, y2], fill = 0)
    elif mode == "erase":
        canvas.create_rectangle(x1, y1, x2, y2, fill = 'white', outline = 'white')
        draw_im.rectangle([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], fill = 255)

# check the direction of an event (slope)
def get_direc(x1, y1, x2, y2):
    ind = (x1 - x2) * (y1 - y2)
    if ind > 0:
        return 1
    if ind < 0:
        return 2
    if x1 == x2:
        return 3
    return 4

# drow lines with thickness
def update_line_with_thick(x1, y1, x2, y2, thickness, canvas):
    half = thickness // 2 + 1
    # half = 1
    direction = get_direc(x1, y1, x2, y2)
    for i in range(half):
        if direction == 1:
            update_line(x1 + i, y1 - i, x2 + i, y2 - i, canvas)
            update_line(x1 - i, y1 + i, x2 - i, y2 + i, canvas)
            update_line(x1, y1 + i, x2, y2 + i, canvas)
            update_line(x1, y1 - i, x2, y2 - i, canvas)
        elif direction == 2:
            update_line(x1 + i, y1 + i, x2 + i, y2 + i, canvas)
            update_line(x1 - i, y1 - i, x2 - i, y2 - i, canvas)
            update_line(x1 + i, y1, x2 + i, y2, canvas)
            update_line(x1 - i, y1, x2 - i, y2, canvas)
        elif direction == 3:
            update_line(x1 + i, y1, x2 + i, y2, canvas)
            update_line(x1 - i, y1, x2 - i, y2, canvas)
        else:
            update_line(x1, y1 + i, x2, y2 + i, canvas)
            update_line(x1, y1 - i, x2, y2 - i, canvas)

# used to check if it is a new point
def distance(x1, y1, x2, y2):
    return (x2 - x1) ** 2 + (y2 - y1) ** 2

# used to change draw or erase mode
def switch_mode(new_mode):
    global mode, erase_button, draw_button
    mode = new_mode
    if mode == "draw":
        draw_button.config(state = "disabled")
        erase_button.config(state = "normal")
    else:
        draw_button.config(state = "normal")
        erase_button.config(state = "disabled")

# used to reset the state when user release the mouse
def reset_position(canvas):
    global last_x, last_y, eraser
    last_x, last_y = None, None
    if eraser:
        canvas.delete(eraser)
# draw main function
def draw(canvas, event):
    global last_x, last_y, eraser, thickness
    x, y = event.x, event.y
    if eraser:
        canvas.delete(eraser)
    if mode == 'draw':
        if last_x:
            update_line_with_thick(last_x, last_y, x, y, thickness, canvas)
            
            update_line_with_thick(last_x, last_y, x, y, thickness, canvas)
        else:
            last_x, last_y = x, y 

    elif mode == 'erase':
        if last_x:
            update_line_with_thick(last_x, last_y, x, y, thickness, canvas)
        else:
            update_line_with_thick(x, y, x + 1, y + 1, thickness, canvas)
        eraser = canvas.create_rectangle(x - thickness // 2 + 1, 
                                         y - thickness // 2 + 1, 
                                         x + thickness // 2 + 1, 
                                         y + thickness // 2 + 1, fill='red', outline = "red")
    last_x, last_y = x, y 


# In[14]:


# thickness function
def update_thickness(val):
    global thickness
    thickness = int(val)


# In[15]:


# clear function
def clear_canvas(canvas, enable_button_lists, disable_button_lists):
    for button in enable_button_lists:
        button.config(state = "normal")
    for button in disable_button_lists:
        button.config(state = "disabled")
    switch_mode("draw")
    canvas.bind("<B1-Motion>", lambda event, canvas = canvas:draw(canvas, event))
    global image, prev_image, image_array, template_photo, draw_im
    canvas.delete("all")
    image = Image.new('L', (resolution, resolution), 255)
    draw_im = ImageDraw.Draw(image)
    prev_image = None
    image_array = None
    template_photo = None


# In[16]:


# template function
# resize the selected images to canvas size
def resize_template(orignal):
    original_shape = max(orignal.shape)
    scale_factor = resolution / original_shape
    square_array = scipy.ndimage.zoom(orignal, scale_factor)
    if orignal.shape[0] == orignal.shape[1]:
        return square_array
    
    top_pad = (resolution - square_array.shape[0]) // 2
    bottom_pad = resolution - square_array.shape[0] - top_pad
    left_pad = (resolution - square_array.shape[1]) // 2
    right_pad = resolution - square_array.shape[1] - left_pad

    # right_pad = resolution - square_array.shape[1]
    # down_pad = resolution - square_array.shape[0]
    square_array = np.pad(square_array, 
                         ((top_pad, bottom_pad), (left_pad, right_pad)),
                         mode = "constant",
                         constant_values = 255)
    # square_array = np.pad(square_array, 
    #                       ((0, down_pad), (0, right_pad)),
    #                       mode = "constant",
    #                       constant_values = 255)
    return square_array

# update the selected image to canvas and cache
def target_template(canvas):
    global image, template_photo, draw_im
    template_path = askopenfilename()
    try:
        template = Image.open(template_path)
        template_array = np.array(template)
        if template_array.ndim > 2:
            raise ValueError("Invalid file")
        template_array = resize_template(template_array)
        template_array = np.minimum(template_array, np.array(image))
        image = Image.fromarray(template_array)
        draw_im = ImageDraw.Draw(image)
        template_photo = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, image = template_photo, anchor = "nw")
    except Exception as e:
        tk.messagebox.showerror("Error", "Failed to load image. Error is %s"%e)


# In[17]:


if __name__ == "__main__":
    resolution = 700
    draw_border = (resolution // 20) ** 2
    eraser, last_x, last_y = None, None, None
    template_photo = None
    image_array = None
    image_size = 128
    transform_edge = transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize(0.5, 0.5),
                           ])
    nz = 100
    ngf = 64
    # model_sub_path = os.path.join("models", "50.pth")
    # if getattr(sys, 'frozen', False):
    #     model_path = os.path.join(sys._MEIPASS, model_sub_path)
    # else:
    #     model_path = os.path.join(model_sub_path)
    # model = Generator(1)
    # model_tmp = torch.load(model_path, map_location = torch.device('cpu'))
    # model.load_state_dict(model_tmp.state_dict())
    model_sub_path = os.path.join("models", "50.pth")
    if getattr(sys, 'frozen', False):
        model_path = os.path.join(sys._MEIPASS, model_sub_path)
    else:
        model_path = os.path.join(model_sub_path) 
    model_tmp = torch.load(model_path, map_location = torch.device('cpu'))
    model = Generator(1)
    model.load_state_dict(model_tmp.state_dict())
    del model_tmp
    mean = 0
    creativity = 1
    thickness = 3
    prev_image = None
    color_balance_range = 50
    main()


# In[18]:


# image_array = np.array(image)


# In[19]:


# Image.fromarray((image_array).astype(np.uint8))


# In[20]:


# image_size = 128
# transform_edge = transforms.Compose([
#                                transforms.Resize(image_size),
#                                transforms.CenterCrop(image_size),
#                                transforms.ToTensor(),
#                                transforms.Normalize(0.5, 0.5),
#                            ])
# image_tensor = transform_edge(Image.fromarray(image_array))


# In[21]:


# model = Generator(1)
# model_path = os.path.join("models", "50.pth")
# # model1_checkpoint = torch.load(model_path, map_location = torch.device('cpu'))
# model_tmp = torch.load(model_path, map_location = torch.device('cpu'))
# model.load_state_dict(model_tmp.state_dict())


# In[22]:


# model.eval()
# input_batch = image_tensor.unsqueeze(0)
# output = model(torch.zeros((1, 100, 1, 1)), input_batch)


# In[23]:


# import torchvision.utils as vutils


# In[24]:


# def tensor2image(tensor):
#     # image = tensor.clone().numpy().transpose(1, 2, 0)
#     low = float(tensor.min())
#     high = float(tensor.max())
#     tensor = tensor.clamp_(min = low, max = high)
#     tensor = tensor.sub_(low).div_(max(high - low, 1e-5))
#     image = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
#     zoom_factor = [resolution / image.shape[0], resolution / image.shape[0], 1]
#     image = scipy.ndimage.zoom(image, zoom_factor)
#     return image


# In[25]:


# def norm_ip(img, low, high):
#     img.clamp_(min=low, max=high)
#     img.sub_(low).div_(max(high - low, 1e-5))
#     return img
# def norm_range(t):
#     return norm_ip(t, float(t.min()), float(t.max()))


# In[26]:


# image = norm_range(output[0])


# In[27]:


# # image = make_grid(output[0], normalize = True)
# image = image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
# zoom_factor = [resolution / image.shape[0], resolution / image.shape[0], 1]
# image = scipy.ndimage.zoom(image, zoom_factor)


# In[28]:


# Image.fromarray(image)


# In[29]:


# vutils.save_image(output[0], "output.png", normalize = True)


# In[30]:


# input_batch = image_tensor.unsqueeze(0)
# nz = 100
# ngf = 64
# mean = 0
# std = 1
# # idx = 8
# fixed_noise = torch.randn(1, nz, 1, 1) * std + mean
# subidx = 2
# # model_lst = ["0.pth", "10.pth", "50.pth", "100.pth", "10000.pth"]
# # model_lst = [os.path.join("version2", "edge2hats.pth")]
# model_lst = ["100.pth", "50.pth"]
# plt.suptitle("$\mu=%d$, $\sigma=%d$"%(mean, std))
# plt.subplot(2, 3, 1)
# plt.imshow(image_tensor.cpu()[0].numpy(), cmap = "gray")
# plt.title("edge")
# # plt.savefig(os.path.join("test_figure", "edge_%s.png"%idx))
# for model_name in model_lst:
#     model_path = os.path.join("models", model_name)
#     # model1_checkpoint = torch.load(model_path, map_location = torch.device('cpu'))
#     model_tmp = torch.load(model_path, map_location = torch.device('cpu'))
#     model.load_state_dict(model_tmp.state_dict())
#     model.eval()
#     input_batch = image_tensor.unsqueeze(0)
#     if torch.cuda.is_available():
#         input_batch = input_batch.cuda()
#         model.to('cuda')
#     with torch.no_grad():
#         output = model(fixed_noise, input_batch)[:, :, []]
#     output_image = tensor2image(output[0])
#     plt.subplot(2, 3, subidx)
#     plt.imshow(output_image)
#     plt.title(model_name)
#     # plt.savefig(os.path.join("test_figure", "%s_%s.png"%(model_name.split(".")[0], idx)))
#     subidx += 1
# # plt.savefig(os.path.join("test_figure", "all_%s.png"%idx))
# plt.show()


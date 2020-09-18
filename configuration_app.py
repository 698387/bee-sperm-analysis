from tkinter import *
from tkinter import font

# The configuration widget
class Configuration(Frame):
    def __init__(self, *args, **kwargs):
        Frame.__init__(self, *args, **kwargs)
        self.create_configuration()
    
    # Create the widget for the configuration
    def create_configuration(self):
        self.configuration_opener = Button(self,
                                           text = "Options...",
                                           command = self.conf_window
                                          )
        self.scale = 0.0
        self.min_length = 0.0
        self.min_movement = 0.0
        self.frame_rate = 0.0
        self.n_frames = 7
        self.view_images = False

        self.window = None
        self.configuration_opener.pack(expand = 1)

    def on_quit(self):
        self.window.destroy()
        self.configuration_opener["state"] = NORMAL
        self.window = None

    # Open the configuration window
    def conf_window(self):
        # It doesn't open a second instance of the window
        if self.window != None:
            return

        self.configuration_opener["state"] = DISABLED
        self.window = Toplevel(self)
        self.window.title("Analyzer options")

        self.window.protocol("WM_DELETE_WINDOW", self.on_quit)

        # Scale of the image
        self.scale_l = Label(self.window, text = "Scale (pixel size)*:")
        self.scale_l.grid(row = 0, sticky = E)
        self.scale_var = DoubleVar()
        self.scale_var.set(self.scale)
        self.scale_var.trace("w", self.on_writing)
        self.scale_e = Entry(self.window, textvariable = self.scale_var)
        self.scale_e.grid(row = 0, column = 1, sticky = E+W)

        # Minimun length
        self.min_len_l = Label(self.window,
                               text = "Minimun spermatozoid length*:")
        self.min_len_l.grid(row = 1, sticky = E)
        self.min_len_var = DoubleVar()
        self.min_len_var.set(self.min_length)
        self.min_len_var.trace("w", self.on_writing)
        self.min_len_e = Entry(self.window, textvariable = self.min_len_var)
        self.min_len_e.grid(row = 1, column = 1, sticky = E+W)

        # Minimun speed
        self.min_speed_l = Label(self.window, text = "Movement threshold*:")
        self.min_speed_l.grid(row = 2, sticky = E)
        self.min_speed_var = DoubleVar()
        self.min_speed_var.set(self.min_movement)
        self.min_speed_var.trace("w", self.on_writing)
        self.min_speed_e = Entry(self.window,
                                 textvariable = self.min_speed_var)
        self.min_speed_e.grid(row = 2, column = 1, sticky = E+W)

        # Frame rate
        self.frame_rate_l = Label(self.window,
                                  text = "Frame rate (frames/seconds):")
        self.frame_rate_l.grid(row = 3, sticky = E)
        self.frame_rate_var = DoubleVar()
        self.frame_rate_var.set(self.frame_rate)
        self.frame_rate_var.trace("w", self.on_writing)
        self.frame_rate_e = Entry(self.window, 
                                  textvariable = self.frame_rate_var)
        self.frame_rate_e.grid(row = 3, column = 1, sticky = E+W)

        # Number of frames to use
        self.n_frames_l = Label(self.window, text = "Frames to use:")
        self.n_frames_l.grid(row = 4, sticky = E)
        self.n_frames_var = IntVar()
        self.n_frames_var.set(self.n_frames)
        self.n_frames_var.trace("w", self.on_writing)
        self.n_frames_e = Entry(self.window, textvariable = self.n_frames_var)
        self.n_frames_e.grid(row = 4, column = 1, sticky = E+W)

        # Show the images of the process
        self.show_frames_var = BooleanVar()
        self.show_frames_var.set(self.view_images)
        self.show_frames_var.trace("w", self.on_writing)
        self.show_frames_c =  Checkbutton(self.window,
                                          variable = self.show_frames_var,
                                          onvalue = True,
                                          offvalue = False,
                                          text = "Show process images")
        self.show_frames_c.grid(row = 5, column = 1, sticky = W)

        # Label of the scale
        self.scale_notation = Label(self.window, text = "* in \u03BCm", 
                                    font = font.Font(slant = font.ITALIC))
        self.scale_notation.grid(row=6, column = 1, sticky = E)

        # Button to save the configuration
        self.save_conf_b = Button(self.window,
                                  text = "Save",
                                  state = DISABLED,
                                  command = self.save_configuration
                                 )
        self.save_conf_b.grid(row = 7, column = 1, sticky = E)

    # Save the configuration given by the user
    def save_configuration(self):
        self.scale = self.scale_var.get()
        self.min_length = self.min_len_var.get()
        self.min_movement = self.min_speed_var.get()
        self.frame_rate = self.frame_rate_var.get()
        self.n_frames = self.n_frames_var.get()
        self.view_images = self.show_frames_var.get()
        self.save_conf_b["state"] = DISABLED

    # Enable the saving button if written any variable
    def on_writing(self, var, idx, mode):
        self.save_conf_b["state"] = NORMAL


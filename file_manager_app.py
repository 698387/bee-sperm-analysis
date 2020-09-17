from tkinter import *
from tkinter import filedialog


videofiletypes = ".mpg .mp2 .mpeg .mpe .mpv .ogg .avi .mov .mp4 .m4p .m4v .wmv .mov .qt"

# Video file manager Widget
class VideoFileManager(Frame):
    def __init__(self, *args, **kwargs):
        Frame.__init__(self, *args, **kwargs)
        self.create_video_manager()

    # Generate the Listbox to manage the videos to analyze
    def create_video_manager(self):
        # Video files storation
        self.video_files = []

        # Adder button
        self.video_adder = Button(self,
                                    text = "Select videos",
                                    command = self.video_selector
                                    )
        self.video_adder.grid(column = 0, row = 2, sticky = N+W)

        # Disadder button
        self.video_deleter = Button(self,
                                    text = "Remove Selected",
                                    command = self.video_deselecter,
                                    state = DISABLED
                                    )
        self.video_deleter.grid(column = 1, row = 2, sticky = N+E)

        # Clear button
        self.video_clear = Button(self,
                                    text = "Clear all videos",
                                    command = self.video_clear,
                                    ).grid(column = 1, row = 3, sticky = N+E)

        # Scrollbars to watch text
        self.vm_x_sb = Scrollbar(self, orient="horizontal")
        self.vm_y_sb = Scrollbar(self, orient="vertical")

        # Listbox showing all video files
        self.video_file_viewer = Listbox(self,
                                            selectmode = MULTIPLE,
                                            xscrollcommand = self.vm_x_sb.set,
                                            yscrollcommand = self.vm_y_sb.set)
        self.vm_x_sb.config(command=self.video_file_viewer.xview)
        self.vm_y_sb.config(command=self.video_file_viewer.yview)
        self.video_file_viewer.grid(column = 0, row = 0, columnspan = 2, sticky = W+N+S+E)
        self.vm_x_sb.grid(column = 0, row = 1, columnspan = 2, sticky = N+W+E)
        self.vm_y_sb.grid(column = 2, row = 0, sticky = N+S+W)
        self.video_file_viewer.bind("<ButtonRelease-1>",
                                    self.on_video_viewer_selection)
        Grid.rowconfigure(self, 0, weight = 1)
        Grid.columnconfigure(self, 0, weight = 1)

    # Add new the videos to analyze
    def video_selector(self):
        # The user select new videos
        new_files = filedialog.askopenfilenames(initialdir = "/", 
                                title = "Select file",
                                filetypes = (("video files", videofiletypes),
                                                ("all files",".*")) )
        # Add the new file to the video files and the listbox
        for new_file in list(new_files):
            self.video_files.append(new_file)
            self.video_file_viewer.insert(END, new_file)

    # Delete the selected video
    def video_deselecter(self):
        deleted_counter = 0
        for delete_idx in list(self.video_file_viewer.curselection()):
            # Delete the video from the list
            self.video_files.pop(delete_idx - deleted_counter)
            # Delete the video from the listbox
            self.video_file_viewer.delete(delete_idx - deleted_counter)
            deleted_counter += 1

    # Clear all the videos
    def video_clear(self):
        self.video_files = []
        self.video_file_viewer.delete(0, END)
        self.video_deleter["state"] = DISABLED

    # Behaviour when user mark videos in the video viewer
    def on_video_viewer_selection(self, event):
        if len(list(self.video_file_viewer.curselection())):
            self.video_deleter["state"] = NORMAL
        else:
            self.video_deleter["state"] = DISABLED

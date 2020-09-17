from tkinter import *
from tkinter import font
from configuration_app import Configuration
from file_manager_app import VideoFileManager
import subprocess as sp
from multiprocessing import Process, Queue
from python_detector.cell_detector import sperm_movility_analysis


class Application(Frame):

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Drone sperm analyzer")
        self.pack(expand = 1, fill = BOTH)
        self.create_widgets()

    # Create the widgets of the app
    def create_widgets(self):
        self.video_manager = VideoFileManager(self)
        self.video_manager.pack(expand=1, fill = BOTH, side = LEFT)
        self.conf = Configuration(self)
        self.conf.pack(expand = 1, fill = BOTH, side = TOP)
        self.analyze = Button(self,
                              text = "Start analyzing",
                              font = font.Font(weight = font.BOLD),
                              command = self.analyze_videos
                              ).pack(expand = 1, side = BOTTOM)


    # Launch the analysis. "self.result_text" widget has to exist
    def launch_analysis(self):
        # Analyze each given video
        for video in self.video_manager.video_files:
            # It uses the configuration values
            result = sperm_movility_analysis(data_file = video,
                                        min_length = self.conf.min_length,
                                        min_movement = self.conf.min_movement,
                                        scale = self.conf.scale,
                                        video_fps = self.conf.frame_rate,
                                        n_frames_to_use = self.conf.n_frames,
                                        view_frames = self.conf.view_images)
            self.result_text.configure(state = NORMAL)  # Enable the text
            # Write the results in the text
            self.result_text.insert(END, "Result of analyzing " + video + "\n")
            for (key, value) in result.items():
                self.result_text.insert(END, "\t" + key + ":")
                self.result_text.insert(END, "\t" + str(value) + "\n")
            self.result_text.configure(state = DISABLED)# Disable the text

    # Analyze all the indicated videos
    def analyze_videos(self):
        self.result_window = Toplevel(self)
        self.result_window.title("Analysis result")
        self.result_text = Text(self.result_window,
                                state = DISABLED)
        self.result_text.pack()

        self.launch_analysis()


def main():
    root = Tk()
    app = Application(master=root)
    app.mainloop()

if __name__ == "__main__":
    main()

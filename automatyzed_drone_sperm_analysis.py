from tkinter import *
from tkinter import font
from tkinter import filedialog
from tkinter.ttk import Progressbar
from configuration_app import Configuration
from file_manager_app import VideoFileManager
import time, threading, queue
from multiprocessing import Process
from python_detector.cell_detector import sperm_movility_analysis
import xlwt


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
                              )
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.analyze.pack(expand = 1, side = BOTTOM)


    # Launch the analysis
    def launch_analysis(video_files, min_lenght, min_movement, scale,
                        frame_rate, n_frames_to_use, view_frames, area_filter):
        all_results = []
        # Analyze each given video
        for video in video_files:
            # It uses the configuration values
            result = sperm_movility_analysis(data_file = video,
                                        min_length = min_lenght,
                                        min_movement = min_movement,
                                        scale = scale,
                                        video_fps = frame_rate,
                                        n_frames_to_use = n_frames_to_use,
                                        view_frames = view_frames,
                                        area_filter = area_filter)

            # Write the results in text
            all_results.append([video, result])

        # Put all the results in the queue
        return all_results

    # Strats the function f with p params in another thread
    def launch_thread(f, args = ()):
        q = queue.Queue()
        t = threading.Thread(target = lambda que, f_args: que.put(f(*f_args)),
                             args = (q, args))
        t.daemon = True
        t.start()
        return t, q

    # Checks if the thread still alive
    def thread_track(self):
        if not self.t.is_alive():
            # Stops the bar
            self.progressbar.stop()
            self.progressbar.destroy()
            self.analyze.pack(expand = 1, side = BOTTOM)
            # Print the results
            self.print_results()
        else:  # Re do the track
            self.after(500, self.thread_track)

    # Show the results inside the queue q
    def print_results(self):
        # Creates a window
        self.result_window = Toplevel(self)
        self.result_window.title("Analysis result")
        # Button for storing the data
        self.save_data = Button(self.result_window,
                                text = "Save Data",
                                command = self.save_result_manager
                                )
        self.save_data.pack()
        # Text widget for writting the result
        self.result_text = Text(self.result_window)
        self.result_text.pack()
        # Get the result from the thread
        self.analysis_result = self.q.get()
        # Prints the result
        for r in self.analysis_result:
            self.result_text.insert(END, r[0] + "\n")
            for key, value in r[1].items():
                self.result_text.insert(END, "\t" + key + ": ")
                self.result_text.insert(END, "\t" + str(value) + "\n ")
        # Disable the writting
        self.result_text["state"] = DISABLED

    # Save result manager, with widget
    def save_result_manager(self):
        # Ask the user for a file
        files = [('Excel files', '*.xls'),  
                    ('All files', '*.*')] 
        file = filedialog.asksaveasfile(filetypes = files,
                                        defaultextension = files)
        # Save the file
        if file != None:
            self.save_result(file.name)

    # Save the analysis result in the file f
    def save_result(self, f = "result.xls"):
        # New excel file
        wb = xlwt.Workbook()
        # Add new sheet
        ws = wb.add_sheet('Analysis Result')
        # Meaning of the values
        ws.write(0, 0, "Video file")
        column_counter = 1
        # No results
        if len(self.analysis_result) == 0:
            wb.save(f)
            return  # exit
        for key in self.analysis_result[0][1].keys():
            ws.write(0, column_counter, key)
            column_counter += 1
        # Each video analyzed
        row_counter = 1
        for result in self.analysis_result:
            # File name
            ws.write(row_counter, 0, result[0])
            column_counter = 1
            # Values from analysis
            for values in result[1].values():
                ws.write(row_counter, column_counter, values)
                column_counter += 1
            row_counter += 1

        # Check the file name extension
        wb.save(f)

    # Analyze all the indicated videos
    def analyze_videos(self):
        # Disables the analyze button
        self.analyze.pack_forget()
        # Launch the process of analyzing
        self.t, self.q = Application.launch_thread(Application.launch_analysis,
                                      args = (self.video_manager.video_files,
                                              self.conf.min_length,
                                              self.conf.min_movement,
                                              self.conf.scale,
                                              self.conf.frame_rate,
                                              self.conf.n_frames,
                                              self.conf.view_images,
                                              self.conf.particles_max_area))
        # Initialize a progress bar
        self.progressbar = Progressbar(self,
                                       orient = HORIZONTAL,
                                       mode='indeterminate')
        self.progressbar.pack(expand = 1, side = BOTTOM)
        self.progressbar.start()
        # Track the thread every 0.5 seconds
        self.after(500, self.thread_track)


    # When the window is destroyed
    def on_closing(self):
        self.conf.write_conf()
        self.master.destroy()

def main():
    root = Tk()
    app = Application(master=root)
    app.mainloop()

if __name__ == "__main__":
    main()

import tkinter as tk
import json
import asyncio
import random
from tkinter import filedialog

class ConversationStream:
    def __init__(self, root):
        self.root = root
        self.root.title("Conversation Stream")
        self.root.configure(bg='black')
        
        # Make it fullscreen
        self.root.attributes('-fullscreen', True)
        self.root.bind('<Escape>', self.exit_fullscreen)
        
        self.conversations = []
        self.current_convo_index = 0
        self.current_message_index = 0
        self.current_char_index = 0
        
        self.streaming = False
        self.skip_requested = False
        self.setup_ui()
        
    def setup_ui(self):
        self.file_button = tk.Button(self.root, text="Upload JSON Files", command=self.load_files, bg='#00ff00', fg='black')
        self.file_button.pack(pady=20)
        
        self.start_button = tk.Button(self.root, text="Start Streaming", command=self.start_streaming, bg='#00ff00', fg='black')
        self.start_button.pack(pady=10)
        
        # Create a frame to hold the text area and center it
        self.frame = tk.Frame(self.root, bg='black')
        self.frame.pack(expand=True, fill='both')
        
        # Calculate the width and height for the text area (80% of screen size)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        text_width = int(screen_width * 0.8)
        text_height = int(screen_height * 0.8)
        
        self.text_area = tk.Text(self.frame, wrap=tk.WORD, bg='black', fg='#00ff00', font=('Courier', 14))
        self.text_area.place(relx=0.5, rely=0.5, width=text_width, height=text_height, anchor='center')
        self.text_area.config(state=tk.DISABLED, padx=20, pady=20)
        
        self.root.bind('<space>', self.request_skip_conversation)
        
    def load_files(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("JSON files", "*.json")])
        for file_path in file_paths:
            with open(file_path, 'r') as file:
                self.conversations.append(json.load(file))
        random.shuffle(self.conversations)
        
    def start_streaming(self):
        if not self.conversations:
            return
        self.file_button.pack_forget()
        self.start_button.pack_forget()
        self.streaming = True
        self.root.after(0, self.run_async_loop)

    def run_async_loop(self):
        asyncio.run(self.stream_conversations())
        
    async def stream_conversations(self):
        while self.streaming:
            await self.stream_conversation(self.conversations[self.current_convo_index])
            if not self.skip_requested:
                self.current_convo_index = (self.current_convo_index + 1) % len(self.conversations)
            self.current_message_index = 0
            self.current_char_index = 0
            self.text_area.config(state=tk.NORMAL)
            self.text_area.delete('1.0', tk.END)
            self.text_area.config(state=tk.DISABLED)
            self.skip_requested = False
            
    async def stream_conversation(self, conversation):
        for message in conversation:
            if self.skip_requested:
                return
            agent = message[1]
            content = message[2]
            await self.stream_message(agent, content)
            self.current_message_index += 1
            self.current_char_index = 0
            await asyncio.sleep(0.3)  # Shorter pause between messages
        
    async def stream_message(self, agent, content):
        agent_color = '#ff69b4' if self.current_message_index % 2 == 0 else '#4169e1'
        self.text_area.config(state=tk.NORMAL)
        self.text_area.insert(tk.END, f"{agent}: ", f"agent_{self.current_message_index}")
        self.text_area.tag_config(f"agent_{self.current_message_index}", foreground=agent_color)
        self.text_area.config(state=tk.DISABLED)
        
        for char in content:
            if self.skip_requested:
                return
            self.text_area.config(state=tk.NORMAL)
            self.text_area.insert(tk.END, char, 'message')
            self.text_area.tag_config('message', foreground='#00ff00')
            self.text_area.config(state=tk.DISABLED)
            self.text_area.see(tk.END)
            self.current_char_index += 1
            await asyncio.sleep(0.01)  # 10ms delay between characters (faster)
            self.root.update()
        
        self.text_area.config(state=tk.NORMAL)
        self.text_area.insert(tk.END, '\n')
        self.text_area.config(state=tk.DISABLED)
        
    def request_skip_conversation(self, event):
        self.skip_requested = True
        self.current_convo_index = (self.current_convo_index + 1) % len(self.conversations)

    def exit_fullscreen(self, event=None):
        self.root.attributes("-fullscreen", False)

if __name__ == "__main__":
    root = tk.Tk()
    app = ConversationStream(root)
    root.mainloop()
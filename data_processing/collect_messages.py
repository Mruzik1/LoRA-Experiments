import discord
import os
import json
import threading
import asyncio
from tqdm.asyncio import tqdm
from dotenv import load_dotenv, find_dotenv
from random import uniform

load_dotenv(find_dotenv())
TOKEN = os.environ.get("DISCORD_TOKEN")


class ChatBot(discord.Client):
    def __init__(self, channel_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channel = channel_id
        self.track_messages = False
        self.pause = False

    def print_bot_info(self):
        print(f"Bot name: {self.user}")
        print(f"Bot ID: {self.user.id}")
        print(f"Bot channel: {self.channel}")
        print(f"Bot tracking messages: {self.track_messages}")

    def print_bot_commands(self):
        print("Bot commands:")
        print("stop - stop the bot")
        print("send <message> - send a message to the channel")
        print("collect <user_id> <output_file> - collect messages from a user")
        print("channel <channel_id> - set the channel to send messages")
        print("track - toggle tracking messages")
        print("help - print the bot commands")
        
    def execute_async_in_thread(self, coro, args):
        loop = asyncio.run_coroutine_threadsafe(coro(*args), self.loop)
        loop.result()
    
    async def execute_stop_cmd(self):
        print("Stopping bot...")
        await self.close()

    async def send_message_cmd(self, msg):
        if not self.channel or not msg.strip():
            print("Unable to send message")
            return
        print(f"Sending message: {msg}")
        channel = self.get_channel(int(self.channel))
        async with channel.typing():
            await asyncio.sleep(uniform(1, 2))
            await channel.send(msg)

    async def collect_messages(self, output_file):
        channel = self.get_channel(int(self.channel))
        history = channel.history(limit=None)
        pbar = tqdm(history, desc=f"Msgs in #{channel.name}", leave=False)
        num_saved = 0

        with open(output_file, "w", encoding="utf-8") as out_f:
            async for message in pbar:
                ref = message.reference
                if not (ref and isinstance(ref.message_id, int)):
                    pbar.set_description(f"[{num_saved}] Skipping - no reference.")
                    continue
                
                try:
                    orig = await channel.fetch_message(ref.message_id)
                except discord.NotFound:
                    pbar.set_description(f"[{num_saved}] Skipping - original message not found.")
                    continue

                num_saved += 1
                pbar.set_description(f"[{num_saved}] Saving the message...")
                record = {
                    "original_user": orig.author.name,
                    "reply_user": message.author.name,
                    "original_id": orig.id,
                    "reply_id": message.id,
                    "original_content": orig.content,
                    "reply_content": message.content,
                    "original_timestamp": orig.created_at.isoformat(),
                    "reply_timestamp": message.created_at.isoformat(),
                    "channel": channel.name
                }

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def set_channel_cmd(self, channel):
        if not channel.isdigit():
            print("Unable to set channel")
            return
        self.channel = channel
        print(f"Channel set to {self.channel}")

    def track_messages_cmd(self):
        self.track_messages = not self.track_messages
        print(f"Tracking messages: {self.track_messages}")
    
    def execute_command(self):
        print("Bot is ready to receive commands")
        while not self.is_closed():
            cmd = input(">>> ").split()
            if not cmd:
                continue
            if cmd[0] == "stop":
                self.execute_async_in_thread(self.execute_stop_cmd, ())
            elif cmd[0] == "help":
                self.print_bot_commands()
            elif cmd[0] == "send":
                self.execute_async_in_thread(self.send_message_cmd, (" ".join(cmd[1:]),))
            elif cmd[0] == "collect":
                self.execute_async_in_thread(self.collect_messages, (cmd[1],))
            elif cmd[0] == "channel":
                self.set_channel_cmd("".join(cmd[1:]))
            elif cmd[0] == "track":
                self.track_messages_cmd()
            else:
                print("Unknown command")

    async def on_ready(self):
        print(f"{self.user} has connected to Discord!")
        self.print_bot_info()
        threading.Thread(target=self.execute_command).start()

    async def on_message(self, msg):
        if str(msg.channel.id) != self.channel:
            return
        if self.track_messages:
            print(f"Message from {msg.author}: {msg.content}\n>>> ", end="")


def main():
    bot = ChatBot("1180845232326180974")
    bot.run(TOKEN)


if __name__ == "__main__":
    main()
import os
import json
import re
import discord
import random
from discord.ext import commands

# config
TOKEN_FILE = 'tokenstorage.json'
DATA_FILE = 'storage.json'

TOKEN = '' 
with open(TOKEN_FILE, 'r') as f:
    TOKEN = json.load(f)["token"]

# setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# --- Helper Functions for JSON Storage ---
def load_links():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_link(link_snippet, msg_id):
    links = load_links()
    links[link_snippet] = msg_id
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(links, f, indent=4)

# event
@bot.event
async def on_ready():
    print(f'{bot.user.name} online now')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
        
    content = message.content

    if "https" in content:
        match = re.search(r'https://[^@\s]+?(/@.*?\?)', content)

        if match:
            unique_data = match.group(1)
            existing_links = load_links()

            if unique_data in existing_links:
                original_msg_id = existing_links[unique_data]
                
                try:
                    original_message = await message.channel.fetch_message(original_msg_id)

                    random_num = random.randint(0, 99)
                    if 0 <= random_num <= 59:
                        await original_message.reply("?")
                    elif 60 <= random_num <= 79:
                        await original_message.reply("<:dog:1233092672835555408>")
                    elif 80 <= random_num <= 99:
                        await original_message.reply("<:frog2:1239954454481076355>")
                        
                except discord.NotFound:
                    await message.channel.send("我找不到是誰傳的，但我很確定有人傳過")
                except discord.HTTPException:
                    await message.channel.send("我找不到是誰傳的，但我很確定有人傳過")
            else:
                save_link(unique_data, message.id)
                print(f"Stored unique snippet: {unique_data}")
        else:
            print("nothing happens")
            pass
            #https but not @? structure

    await bot.process_commands(message)

bot.run(TOKEN)
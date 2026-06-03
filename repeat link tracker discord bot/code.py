import os
import json
import re
import discord
import random
import aiohttp
from discord import app_commands

from discord.ext import commands
  
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta

DATA_FILE = 'storage.json'
TOKEN_FILE = 'tokenstorage.json'

if os.environ.get("TOKEN"):
    TOKEN = os.environ.get("TOKEN")
else:
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'r') as f:
            TOKEN = json.load(f)["token"]
    else:
        TOKEN = "TOKENNOTFOUND"

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

noTimestored = True
first_stored_time = "NA"

last_message_time = None  
USER_MAPPING = {
    "userID_1": "userName_1",
    "userID_2": "userName_2",
    "userID_3": "userName_3",
    "userID_4": "userName_4",
    "userID_5": "userName_5",
    "userID_6": "userName_6"
}

today_counters = {name: 0 for name in USER_MAPPING.values()}
yesterday_counters = {name: 0 for name in USER_MAPPING.values()}

total_repeat_counters = {name: 0 for name in USER_MAPPING.values()}

def load_links():
    # Input: None
    # Output: {str: int} — link_snippet → message_id; {} if file absent
    # Utility: Reads and returns the persisted link storage dict from disk.
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_link(link_snippet, msg_id):
    # Input: link_snippet(str), msg_id(int)
    # Output: None (writes to disk)
    # Utility: Appends a new link snippet + message ID pair to storage.json.
    links = load_links()
    links[link_snippet] = msg_id
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(links, f, indent=4)

async def get_bilibili_bvid_async(short_url: str) -> str | None:
    # Input: short_url(str)
    # Output: str | None — BV ID e.g. "BV1xx411c7mD", or None on failure
    # Utility: Resolves a b23.tv short URL via HTTP HEAD redirect and extracts the Bilibili BV ID.
    # Exception Handler:
    # - Exception: catches all network/timeout errors from aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.head(short_url, allow_redirects=False, timeout=5) as response:
                if response.status in [301, 302]:
                    real_url = response.headers.get('Location', '')
                    bv_match = re.search(r'(BV[a-zA-Z0-9]{10})', real_url)
                    if bv_match:
                        return bv_match.group(1)
        return None
    except Exception as e:
        print(f"[Bilibili Async Error]: {e}")
        return None


async def extract_link_info(content: str) -> str:
    # Input: content(str)
    # Output: str — "NOTLINK" | "/@<path>?" | "/reel/<id>?" | "BILI_BV<id>"
    # Utility: Parses a Discord message and extracts a platform-specific unique identifier
    #          from Threads, Instagram Reels, or Bilibili links.
    if "https" not in content:
        return "NOTLINK"
        
    if "threads" in content:
        match = re.search(r'https://[^@\s]+?(/@.*?\?)', content)
        if match:
            return match.group(1)
            
    if "instagram" in content:
        match = re.search(r'/reel/([^?\s]+\?)', content)
        if match:
            return match.group(1)

    if "b23.tv" in content:
        match = re.search(r'b23\.tv/([a-zA-Z0-9]{7})', content)
        if match:
            clean_short_url = f"https://b23.tv/{match.group(1)}"
            bvid = await get_bilibili_bvid_async(clean_short_url)
            if bvid:
                return f"BILI_{bvid}"

    if "bilibili.com/video/" in content:
        match = re.search(r'bilibili\.com/video/(BV[a-zA-Z0-9]{10})', content)
        if match:
            return f"BILI_{match.group(1)}"

    return "NOTLINK"

def check_and_update_epoch(current_msg_utc):
    # Input: current_msg_utc(datetime) — UTC-aware datetime
    # Output: None (mutates globals: last_message_time, today_counters, yesterday_counters)
    # Utility: Detects 05:00 Taipei-time day boundary and rolls today's counters into
    #          yesterday's, resetting today's to zero on rollover.
    global last_message_time, today_counters, yesterday_counters
    
    current_taipei = current_msg_utc.astimezone(ZoneInfo("Asia/Taipei"))    
    if last_message_time is None:
        last_message_time = current_taipei
        return

    boundary = current_taipei.replace(hour=5, minute=0, second=0, microsecond=0)
    if current_taipei < boundary:
        boundary -= timedelta(days=1)

    if last_message_time < boundary:
        yesterday_counters = today_counters.copy()
        today_counters = {name: 0 for name in USER_MAPPING.values()}
        print(f"Epoch Rollover 5:00 AM boundary")

    last_message_time = current_taipei

@bot.event
async def on_ready():
    # Input: None (Discord event)
    # Output: None (prints to stdout; syncs slash commands to Discord)
    # Utility: Fires on bot login; registers all slash commands globally via tree.sync().
    # Exception Handler:
    # - Exception: catches Discord API sync failures
    print(f'{bot.user.name} online now')
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} slash commands successfully!")
    except Exception as e:
        print(f"Failed to sync commands: {e}")

@bot.tree.command(name="info", description="Overview of storage.json")
async def info_command(interaction: discord.Interaction):
    # Input: interaction(discord.Interaction)
    # Output: None (sends Discord message with link count and optional earliest timestamp)
    # Utility: /info slash command — reports total stored links and earliest-stored timestamp of current session.
    links = load_links()
    
    storage_size = len(links)
    response_msg = f"儲存的鏈接數量: {storage_size}"
    
    if noTimestored == True:
        pass
    else:
        response_msg += f"\n最早的鏈接發送時間: {first_stored_time}"
        
    await interaction.response.send_message(response_msg)

@bot.tree.command(name="today", description="顯示今日計數")
async def today_slash(interaction: discord.Interaction):
    # Input: interaction(discord.Interaction)
    # Output: None (sends multi-line Discord message)
    # Utility: /today slash command — displays each tracked user's unique link count for the current day epoch.
    lines = ["今日統計:"]
    for user, count in today_counters.items():
        lines.append(f"• {user}: {count}")
    await interaction.response.send_message("\n".join(lines))

@bot.tree.command(name="yesterday", description="顯示昨日計數")
async def yesterday_slash(interaction: discord.Interaction):
    # Input: interaction(discord.Interaction)
    # Output: None (sends multi-line Discord message)
    # Utility: /yesterday slash command — displays each tracked user's unique link count for the previous day epoch.
    lines = ["昨日統計:"]
    for user, count in yesterday_counters.items():
        lines.append(f"• {user}: {count}")
    await interaction.response.send_message("\n".join(lines))

@bot.tree.command(name="repeat", description="顯示重複計數")
async def repeat_slash(interaction: discord.Interaction):
    # Input: interaction(discord.Interaction)
    # Output: None (sends multi-line Discord message)
    # Utility: /repeat slash command — displays all-time duplicate link counts per tracked user.
    lines = ["重複統計:"]
    for user, count in total_repeat_counters.items():
        lines.append(f"• {user}: {count}")
    await interaction.response.send_message("\n".join(lines))

@bot.tree.command(name="export", description="Export current storage.json")
async def export_slash(interaction: discord.Interaction):
    # Input: interaction(discord.Interaction)
    # Output: None (sends storage.json as a timestamped Discord file attachment)
    # Utility: /export slash command — exports the current link-storage JSON as a downloadable attachment.
    # Exception Handler:
    # - Exception: sends ephemeral error message on failure
    if not os.path.exists(DATA_FILE):
        await interaction.response.send_message("Cannot find storage.json.", ephemeral=True)
        return

    await interaction.response.defer()

    try:
        current_time_taipei = datetime.now(ZoneInfo("Asia/Taipei"))
        dynamic_filename = current_time_taipei.strftime("%Y%m%d_%H%M%S.json")

        discord_file = discord.File(DATA_FILE, filename=dynamic_filename)
        await interaction.followup.send(f"Current storage.json: ", file=discord_file)
    except Exception as e:
        await interaction.followup.send(f"Exception：{e}", ephemeral=True)

@bot.tree.command(name="import", description="import json file for storage.json")
async def import_slash(interaction: discord.Interaction, file: discord.Attachment):
    # Input: interaction(discord.Interaction), file(discord.Attachment)
    # Output: None (overwrites storage.json; sends ephemeral status message)
    # Utility: /import slash command (Plonese-only) — replaces storage.json with an uploaded .json file.
    # Exception Handler:
    # - json.JSONDecodeError: invalid JSON format in uploaded file
    # - Exception: any other read/write error
    if str(interaction.user.id) != "userID_1":
        await interaction.response.send_message("Only Plonese can use this command.", ephemeral=True)
        return

    if not file.filename.endswith('.json'):
        await interaction.response.send_message("json file required", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True)

    try:
        file_bytes = await file.read()
        new_data = json.loads(file_bytes.decode('utf-8'))

        if not isinstance(new_data, dict):
            await interaction.followup.send("invalid Dict format", ephemeral=True)
            return

        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=4)

        await interaction.followup.send(f"reading {len(new_data)} data successfully", ephemeral=True)
        print(f"Admin Log import json ({file.filename})")

    except json.JSONDecodeError:
        await interaction.followup.send("Invalid json format", ephemeral=True)
    except Exception as e:
        await interaction.followup.send(f"Exception：{e}", ephemeral=True)

@bot.event
async def on_message(message):
    # Input: message(discord.Message) (Discord event)
    # Output: None (side effects: updates counters, saves links, sends Discord replies)
    # Utility: Core event handler — detects duplicate links, replies with a random reaction
    #          to the original poster, and tracks per-user unique link counts.
    # Exception Handler:
    # - discord.NotFound: original message no longer exists
    # - discord.HTTPException: Discord API failure when fetching message
    global noTimestored, first_stored_time, today_counters, total_repeat_counters

    if message.author == bot.user:
        return
        
    check_and_update_epoch(message.created_at)

    content = message.content
    unique_data = await extract_link_info(content)

    if unique_data != "NOTLINK":
        existing_links = load_links()

        if unique_data in existing_links:
            original_msg_id = existing_links[unique_data]
            
            author_id_str = str(message.author.id)
            if author_id_str in USER_MAPPING:
                username = USER_MAPPING[author_id_str]
                total_repeat_counters[username] += 1

            try:
                original_message = await message.channel.fetch_message(original_msg_id)
                random_num = random.randint(0, 99)
                if 0 <= random_num <= 39:
                    await original_message.reply("?")
                elif 40 <= random_num <= 69:
                    await original_message.reply("<:dog:1233092672835555408>")
                elif 70 <= random_num <= 99:
                    await original_message.reply("<:frog2:1239954454481076355>")
                        
            except discord.NotFound:
                await message.channel.send("我找不到是誰傳的，但我很確定有人傳過")
            except discord.HTTPException:
                await message.channel.send("我找不到是誰傳的，但我很確定有人傳過")
            
        else:
            if noTimestored:
                noTimestored = False
                local_time = message.created_at.astimezone(ZoneInfo("Asia/Taipei"))
                first_stored_time = local_time.strftime('%Y-%m-%d %H:%M:%S')
                print(f"[Runtime Log] First link caught since restart at: {first_stored_time}")

            save_link(unique_data, message.id)
            print(f"Stored unique snippet: {unique_data}")

            author_id_str = str(message.author.id)
            if author_id_str in USER_MAPPING:
                username = USER_MAPPING[author_id_str]
                today_counters[username] += 1

    await bot.process_commands(message)

bot.run(TOKEN)

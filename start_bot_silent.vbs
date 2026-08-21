Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "C:\Users\Kiongozi Legit\Downloads\websites\mjbot"
WshShell.Run "cmd /c python -u bot.py >> bot.log 2>> bot_err.log", 0, False

import json, os, keyboard

def data_generator(folder_path="data", json_prefix="sentiment_analysis_data-", num_jsons=50, start_index=1): 
    for n in range(start_index, start_index + num_jsons):
        with open(os.path.join(folder_path, json_prefix + str(n) + ".json")) as f:
            for line in f:
                entry = json.loads(line)
                yield entry


if __name__ == "__main__":
    print("testing data generator")
    entry_gen = data_generator()
    keyboard.add_hotkey("enter", lambda : print(next(entry_gen)), suppress=True)
    print("Press enter to print next entry, press esc to leave.")
    keyboard.wait('esc', suppress=True)
    keyboard.remove_all_hotkeys()
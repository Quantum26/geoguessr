import json, os, sys, keyboard, csv, re, argparse

def data_generator(folder_path="data", json_prefix="sentiment_analysis_data-", num_jsons=50, start_index=1): 
    for n in range(start_index, start_index + num_jsons):
        with open(os.path.join(folder_path, json_prefix + str(n) + ".json")) as f:
            for line in f:
                entry = json.loads(line)
                yield entry

def generate_line_list(gen=None, folder_path="data", save=False, load=False):
    '''
    Function to create a list of word groupings for Word2Vec and Model Training.
    Takes a generator/iterator of dictionaries such as:
        [{"label" : 0, "data" : "This is a sentence."},
         {"label" : 1, "data" : "I am also a sentence. Or am I?"}]
    And creates the following list structure:
        [["this", "is", "a", "sentence"],
         ["i", "am", "also", "a", "sentence", "or", "am", "i"]]
    
    This function should convert the sentences to all lowercase and remove special characters.
    '''
    text_list = []
    file_path = os.path.join(folder_path, "training_text.csv")

    if load:
        if not os.path.isfile(file_path):
            raise FileNotFoundError("Run 'python data_processing.py --save' first before you load a csv that doesn't exist.")
        with open(file_path, 'r', newline='') as f:
            for line in csv.reader(f):
                text_list.append(line)
        return text_list
    
    if gen is None:
        raise Exception("Please use a valid generator/iterator for line list generation.")
    
    if save:
        f = open(file_path, 'w', newline='')
        writer = csv.writer(f)
    for entry in gen:
        line = list(filter(None, re.sub(r'[^a-zA-Z0-9\s]+', '', entry["data"].replace("\n", "")).lower().split(' ')))
        text_list.append(line)
        if save:
            writer.writerow(line)
    if save:
        f.close()

    return text_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action='store_true', help="Test saving line list.")
    parser.add_argument("--load", action='store_true', help="Test loading line list.")
    parser.set_defaults(save=False)
    parser.set_defaults(load=False)
    args = parser.parse_args()

    if args.save:
        print("Saving line list as csv")
        entry_generator = data_generator()
        generate_line_list(entry_generator, save=True)
        print("Line list saved in 'data/training_text.csv'")

    if args.load:
        print("testing loaded sentence list")
        def sentence_gen():
            for entry in generate_line_list(gen=None, load=True):
                yield entry
        entry_gen = sentence_gen()
        keyboard.add_hotkey("enter", lambda : print(next(entry_gen)), suppress=True)
        print("Press enter to print next entry, press esc to leave.")
        keyboard.wait('esc', suppress=True)
        keyboard.remove_all_hotkeys()

    elif not args.save:
        print("testing data generator")
        entry_gen = data_generator()
        keyboard.add_hotkey("enter", lambda : print(next(entry_gen)), suppress=True)
        print("Press enter to print next entry, press esc to leave.")
        keyboard.wait('esc', suppress=True)
        keyboard.remove_all_hotkeys()
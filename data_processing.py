import json, os, sys, keyboard, csv, re, argparse

def data_generator(folder_path="data", json_prefix="sentiment_analysis_data-", num_jsons=50, start_index=1, debug=False): 
    for n in range(start_index, start_index + num_jsons):
        if debug:
            print("Opening " + json_prefix + str(n) + ".json")
        with open(os.path.join(folder_path, json_prefix + str(n) + ".json")) as f:
            for line in f:
                entry = json.loads(line)
                yield entry
        if debug:
            print("Closed " + json_prefix + str(n) + ".json")

def line_list_generator(gen=None, folder_path="data", save=False, load=False, sparse=False, max_files=50):
    '''
    Function to create a list of word groupings for Word2Vec and Model Training.
    Takes a generator/iterator of dictionaries such as:
        [{"label" : 0, "data" : "This is a sentence."},
         {"label" : 1, "data" : "I am also a sentence. Or am I?"}]
    And creates the following list structure:
        [["this", "is", "a", "sentence"],
         ["i", "am", "also", "a", "sentence", "or", "am", "i"]]
    
    This function should convert the sentences to all lowercase and remove special characters.

    For save function, will limit .csv files to 100000 "lines" each.
    '''
    file_path = os.path.join(folder_path, "training_text-")

    if load:
        def lines_gen():
            for n in range(1, max_files+1):
                line_list=[]
                if not os.path.isfile(file_path + str(n) + ".csv"):
                    break
                with open(file_path + str(n) + ".csv", 'r', newline='') as f:
                    for line in csv.reader(f):
                        line_list.append(line)
                yield line_list
        return lines_gen()
    
    if gen is None:
        raise Exception("Please use a valid generator/iterator for line list generation.")
    def lines_gen():
        f=None
        writer=None
        entries = 100000
        for i, entry in enumerate(gen):
            if i%entries == 0:
                text_list = []
                if i>=entries:
                    if not sparse:
                        yield text_list
                    if save:
                        f.close()
                if save:
                    f = open(file_path  + str(int(i/entries)+1) + ".csv", 'w', newline='')
                    writer = csv.writer(f)

            line = list(filter(None, re.sub(r'[^a-zA-Z0-9\s]+', ' ', entry["data"].replace("\n", " ").encode('ascii', 'ignore').decode('ascii')).lower().split(' ')))
            
            if not sparse:
                text_list.append(line)
            if save:
                writer.writerow(line)
    return lines_gen()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action='store_true', help="Test saving line list.")
    parser.add_argument("--load", action='store_true', help="Test loading line list.")
    parser.set_defaults(save=False)
    parser.set_defaults(load=False)
    args = parser.parse_args()

    if args.save:
        print("Saving line list as csv")
        entry_generator = data_generator(num_jsons=70, debug=True)
        line_generator = line_list_generator(entry_generator, save=True, sparse=True)
        list(line_generator)
        print("Line list saved in 'data/training_text.csv'")

    if args.load:
        print("testing loaded sentence list")
        entry_gen = line_list_generator(load=True)
        keyboard.add_hotkey("enter", lambda : print(len(next(entry_gen))), suppress=True)
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
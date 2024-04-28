
import pickle


def load_stub(stub_path):
    try:
        with open(stub_path, 'rb') as f:
            tracks = pickle.load(f)
        return tracks
    except Exception as e:
        print("Error reading stub: ", e)


def save_stub(data, stub_path):
    try:
        with open(stub_path, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print("Error saving stub: ", e)

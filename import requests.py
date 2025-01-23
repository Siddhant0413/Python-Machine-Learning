import requests
import pandas as pd

def decode_secret_message(url):
    """
    Fetches data from the given URL, parses it into a grid, 
    and prints the secret message.

    :param url: The URL of the Google Doc containing the input data.
    """
    # Fetch data from URL (mocked for now, replace with actual logic)
    # response = requests.get(url)
    # content = response.text

    # Mocked data based on the provided image
    data = [
        {'x-coordinate': 0, 'Character': '█', 'y-coordinate': 0},
        {'x-coordinate': 0, 'Character': '█', 'y-coordinate': 1},
        {'x-coordinate': 0, 'Character': '█', 'y-coordinate': 2},
        {'x-coordinate': 1, 'Character': '█', 'y-coordinate': 1},
        {'x-coordinate': 1, 'Character': '█', 'y-coordinate': 2},
        {'x-coordinate': 2, 'Character': '█', 'y-coordinate': 1},
        {'x-coordinate': 2, 'Character': '█', 'y-coordinate': 2},
        {'x-coordinate': 3, 'Character': '█', 'y-coordinate': 2},
    ]

    # Convert to DataFrame for easy processing
    df = pd.DataFrame(data)

    # Determine the dimensions of the grid
    max_x = df['x-coordinate'].max()
    max_y = df['y-coordinate'].max()

    # Create an empty grid filled with spaces
    grid = [[' ' for _ in range(max_x + 1)] for _ in range(max_y + 1)]

    # Populate the grid with characters from the data
    for _, row in df.iterrows():
        x, y, char = row['x-coordinate'], row['y-coordinate'], row['Character']
        grid[y][x] = char

    # Print the grid row by row
    for row in grid:
        print(''.join(row))

# Example usage
mock_url = "https://docs.google.com/document/d/e/2PACX-1vQGUck9HIFCyezsrBSnmENk5ieJuYwpt7YHYEzeNJkIb9OSDdx-ov2nRNReKQyey-cwJOoEKUhLmN9z/pub"  # Replace with actual URL if needed
decode_secret_message(mock_url)

# Creating a Python script that reads text files and splits them into chunksFile I/O, string manipulation

def split_file_into_chunks(file_path, chunk_size):
    """
    Splits the content of a text file into chunks of specified size.

    :param file_path: Path to the text file to be read.
    :param chunk_size: The size of each chunk in characters.
    :return: A list of chunks.
    """
    chunks = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            for i in range(0, len(content), chunk_size):
                chunks.append(content[i:i + chunk_size])
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return chunks

def main():
    file_path = input("Enter the path to the text file: ")
    chunk_size = int(input("Enter the chunk size (number of characters): "))
    
    chunks = split_file_into_chunks(file_path, chunk_size)
    
    for index, chunk in enumerate(chunks):
        print(f"Chunk {index + 1}:\n{chunk}\n{'-' * 40}")


if __name__ == "__main__":
    main()
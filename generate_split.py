from data_loader import generate_and_save_split
if __name__ == '__main__':
    generate_and_save_split()
    print('\nDone.  You can now run:  python train.py --model <name> --run <1|2|3>')

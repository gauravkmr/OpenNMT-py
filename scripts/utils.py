
def main():
    max_size = 0
    x = ""
    with open("/Users/gaurav/spring2019/RA/projects/bilingual/OpenNMT-py/data/zul_train_LRLP_TOKENIZED_SOURCE.txt") as f:
        for line in f:
            if len(line.strip()) > max_size:
                max_size = len(line.strip())
                x = line.strip()

    print(max_size)
    print(x)

if __name__ == '__main__':
    main()
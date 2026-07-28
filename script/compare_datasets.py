import json
from argparse import ArgumentParser
from tqdm import tqdm



def compare_dataset(old, new):
    with open(old,'r', encoding="utf-8") as fin1:
        with open(new,'r', encoding="utf-8") as fin2:
            all_old = {}
            for line in fin1:
                line = json.loads(line)
                all_old[line['doc_id']+' '+line['sent_id']] = line
            all_new = {}
            for line in fin2:
                line = json.loads(line)
                all_new[line['doc_id']+' '+line['sent_id']] = line
            
            print(len(all_old),len(all_new))

            num_problem = 0
            for key, value in all_new.items():
                if not key in all_old:
                    print('missing',value)
                    num_problem+=1
                else:
                    if not value == all_old[key]:
                        print('-------------------------------')
                        print('old  ',all_old[key])
                        print('new  ', value)
                        num_problem+=1
            print(num_problem)

            # for line1, line2 in zip(fin1,fin2):
            #     line1 = json.loads(line1)
            #     line2 = json.loads(line2)
            #     if not line1 == line2:
            #         print('---------------------------------------------------')
            #         print('old  ',line1)
            #         print('new  ', line2)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', default='./data/ere_oneie/test.oneie.json')
    parser.add_argument('-o', '--output', default='./data/ere_amrie/test.oneie.json')
    
    args = parser.parse_args()
    compare_dataset(args.input, args.output)
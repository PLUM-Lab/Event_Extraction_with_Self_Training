import random

lrs = ['0.001']
blrs = ['0.0001']
mix_ratios = ['1.0']
# feedbacks = [True, False]
feedbacks = [True]
aux2trains = ['2','4','8','12']
# aux2trains = ['4','8','12','14']
data_list = ['ere']
# data_list = ['ere']
exp_idx_list = ['1','2','3','4','5']
num_random_seeds = 1

for a in aux2trains:
    for f in feedbacks:
        name = 'self_seq_ere_' if f else 'self_ere_'
        with open("shell_command/run_"+ name +a+'.sh','w') as fin:
            for m in mix_ratios:
                for lr in lrs:
                    for blr in blrs:
                        if float(blr) > float(lr):
                            continue
                        for exp_idx in exp_idx_list:
                            for data in data_list:
                                random_seeds = []
                                while len(random_seeds)<num_random_seeds:
                                    seed = random.randint(0, 10000)
                                    if seed in random_seeds:
                                        continue
                                    if data == 'ACE':
                                        config_file = 'baseline.json'
                                    elif data == 'ACE+':
                                        config_file = 'baseline_ace+.json'
                                    elif data == 'ere':
                                        config_file = 'baseline_ere.json'  
                                    command="python self_train.py -c config/{} -n {}/{}/wo_trigger/lr-{}_blr-{}_m-{}_a-{}_final-{}_seed-{} -g 0 -s {} -l -lr {} -blr {} -m {} -a {} -i {} {}".format(config_file,data,'self_seq' if f else 'self',lr,blr,m,a,exp_idx,seed,seed,lr,blr,m,a, exp_idx,"-f" if f else "")
                                    fin.write(command+'\n\n')
                                    random_seeds.append(seed)

